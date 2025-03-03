from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh

from cornstarch.pipeline_template import PipelineTemplate


class MultiModalProcessGroupMesh(ProcessGroupMesh):
    """
    A helper class to manage the process group mesh.

    We use a ND-tuple to represent the process group mesh,
    and a ND-coordinate is to represent each process.
    For example, ``(0, 1, 0)`` represents the process whose rank is 2 in
    a 3D process group mesh with size ``(2, 2, 2)``.

    Different from the original `ProcessGroupMesh`, `MultiModalProcessGroupMesh`
    takes multiple modal templates and execution order as input, and creates
    a single unified mesh for the glued multimodal model.

    Args:
        modal_templates (dict[PipelineTemplate, int]): The modal templates and their tp sizes.
            Each modal may have different number of stages and different tp sizes.
        llm_template (tuple[PipelineTemplate, int, int]): The LLM template, tp size, and sp size.
        execution_order (list[tuple[PipelineTemplate, PipelineTemplate]]): The execution order of the modals.
            `MultiModalProcessGroupMesh` uses topological sort to determine the order of the modals.
            This is not related to actual execution order, but only used to assign ranks to modal models.
    """

    pp_axis, dp_axis, sp_axis, tp_axis = 0, 1, 2, 3

    def __init__(
        self,
        encoder_templates: Optional[dict[PipelineTemplate, int]] = None,
        llm_template: Optional[tuple[PipelineTemplate, int, int]] = None,
        decoder_templates: Optional[dict[PipelineTemplate, int]] = None,
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."

        if encoder_templates is None:
            encoder_templates = {}

        if decoder_templates is None:
            decoder_templates = {}
        if len(decoder_templates) > 0:
            assert (
                llm_template is not None
            ), "LLM template is required if decoders are given."

        self.encoder_templates = encoder_templates
        self.llm_template = llm_template
        self.decoder_templates = decoder_templates
        self.topological_sorted_modals: list[PipelineTemplate] = [
            *self.encoder_templates.keys(),
            *([] if llm_template is None else [llm_template[0]]),
            *self.decoder_templates.keys(),
        ]

        llm_tp_size, llm_sp_size = llm_template[1], llm_template[2]
        num_ranks_in_model = (
            sum(
                template.num_stages * tp_size
                for template, tp_size in encoder_templates.items()
            )
            + (
                0
                if llm_template is None
                else llm_template[0].num_stages * llm_tp_size * llm_sp_size
            )
            + sum(
                template.num_stages * tp_size
                for template, tp_size in decoder_templates.items()
            )
        )
        assert (
            dist.get_world_size() % num_ranks_in_model == 0
        ), f"The world size {dist.get_world_size()} must be divisible by num_ranks per replica {num_ranks_in_model}."
        dp_size = dist.get_world_size() // num_ranks_in_model

        max_tp_size = max(
            [
                *encoder_templates.values(),
                0 if llm_template is None else llm_tp_size,
                *decoder_templates.values(),
            ]
        )
        meshes: list[list[list[int]]] = []
        rank_index = 0
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)

        for iterable in [
            encoder_templates.items(),
            [llm_template[:2]],
            decoder_templates.items(),
        ]:
            for modal, tp_size in iterable:
                for _ in range(modal.num_stages):
                    stage_mesh = []
                    for _ in range(dp_size):
                        # LLM may have context parallelism,
                        # in that case different ranks must be assigned.
                        if modal == llm_template[0]:
                            tp_mesh = []
                            for _ in range(llm_sp_size):
                                ranks = [
                                    i
                                    for i in range(rank_index, rank_index + tp_size)
                                    for _ in range(max_tp_size // tp_size)
                                ]
                                rank_index += tp_size
                                tp_mesh.append(ranks)
                        else:
                            # create a list of ranks with length `max_tp_size`, where each rank is repeated `max_tp_size // tp_size` times.
                            # Example: [0, 0, 1, 1] for tp_size=2 and max_tp_size=4
                            ranks = [
                                i
                                for i in range(rank_index, rank_index + tp_size)
                                for _ in range(max_tp_size // tp_size)
                            ]
                            rank_index += tp_size
                            tp_mesh = [ranks for _ in range(llm_sp_size)]

                        stage_mesh.append(tp_mesh)
                        modal_to_ranks[modal].extend(list(itertools.chain(*tp_mesh)))
                    meshes.append(stage_mesh)

        self._rank = dist.get_rank()
        self._mesh = np.array(meshes)
        self._shape = self._mesh.shape

        self._coords = MultiModalProcessGroupMesh.unravel(self._rank, self._mesh)
        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}

        self.modal_to_ranks = {
            modal: list(set(ranks)) for modal, ranks in modal_to_ranks.items()
        }

    @property
    def coords(self) -> list[tuple[int, ...]]:
        """The process coordinates.

        Returns:
            list[tuple[int, ...]]: The process coordinates.
        """
        return self._coords

    @property
    def mesh(self) -> np.ndarray:
        """The process rank mesh.

        Returns:
            np.ndarray: The process rank mesh.
        """
        return self._mesh

    @staticmethod
    def unravel(rank: int, mesh: np.ndarray) -> list[tuple[int, ...]]:
        """Convert a rank to a list of coordinates.

        Unlike colossalai.cluster.process_group_mesh.ProcessGroupMesh.unravel,
        our mesh manages process groups per layer; hence the same rank can exist
        in multiple coordinates.

        Args:
            rank (int): Rank to be converted.
            mesh (tuple[int, ...]): A grid of process ranks.

        Returns:
            list[tuple[int, ...]]: List of coordinates of the rank.
        """
        indices = np.where(mesh == rank)
        return list(zip(*indices))

    def create_or_get_group_along_axis(
        self,
        axis: int | list[int],
        indices_at_axis: list[int] | list[list[int]],
        target_ranks_in_group: tuple[int],
        backend: Optional[str] = None,
    ) -> dist.ProcessGroup:
        indices_at_axis = indices_at_axis or [
            list(range(self._shape[ax])) for ax in axis
        ]
        reduced_shape = list(self._shape)
        # the choices on the axis are reduced to 1, since it's determined by `indices_at_axis`
        for ax in axis:
            reduced_shape[ax] = 1
        target_group = None
        # use Cartesian product to generate all combinations of coordinates
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(
                base_coord, axis, indices_at_axis
            )
            ranks_in_group = tuple(
                sorted(set([self._mesh[coord] for coord in coords_in_group]))
            )

            group = self._get_group(ranks_in_group, backend=backend)
            if target_ranks_in_group == ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self,
        axis: int | list[int],
        indices_at_axis: Optional[list[int] | list[list[int]]] = None,
        backend: Optional[str] = None,
    ) -> dist.ProcessGroup | list[dist.ProcessGroup]:
        """Get the process group along the given axis which the current process belongs to.
        If the process group doesn't exist, it will be created.

        A rank may exist multiple times in the mesh as modals may have different number of stages and tp sizes.
        If `axis` is dp_axis, no matter how many times a rank exists in the mesh,
        it should belong to the same dp group, thus return a single `dist.ProcessGroup`.
        If `axis` is pp_axis, a rank may belong to multiple pp groups, thus return a list of `dist.ProcessGroup`.

        Args:
            axis (int): The axis along which the group is created.
            indices_at_axis (list[int], optional): The indices at the axis. Defaults to None.
            backend (str, optional): The backend to create the group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
            list[ProcessGroup]: if `axis` == pp_axis, a single rank may belong to multiple pp groups.
                In such case, a list of process groups will be returned.
        """
        if isinstance(axis, int):
            axis = [axis]
            if indices_at_axis is not None:
                assert isinstance(indices_at_axis[0], int)
                indices_at_axis = [indices_at_axis]

        if MultiModalProcessGroupMesh.pp_axis in axis:
            assert len(axis) == 1, "Only one axis is allowed for pp group."

        indices_at_axis = indices_at_axis or [
            list(range(self._shape[ax])) for ax in axis
        ]

        reduced_shape = list(self._shape)
        for ax in axis:
            reduced_shape[ax] = 1

        process_group_list: list[dist.ProcessGroup] = []
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = self.get_coords_along_axis(
                base_coord, axis, indices_at_axis
            )

            ranks_in_group = tuple(
                sorted(set([self._mesh[coord] for coord in coords_in_group]))
            )
            group = self.create_or_get_group_along_axis(
                axis, indices_at_axis, ranks_in_group, backend
            )

            if self._rank in ranks_in_group:
                process_group_list.append(group)

        process_group_list = list(set(process_group_list))

        if len(process_group_list) > 1 or axis[0] == MultiModalProcessGroupMesh.pp_axis:
            return process_group_list
        else:
            return process_group_list[0]
