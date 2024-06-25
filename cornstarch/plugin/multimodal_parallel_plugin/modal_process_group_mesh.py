from __future__ import annotations

import copy
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh

from cornstarch.pipeline_template import PipelineTemplate


@dataclass
class ModalDependencies:
    modal: PipelineTemplate
    in_degree: int
    out_degree: int
    previous: list[PipelineTemplate]
    next: list[PipelineTemplate]

    @classmethod
    def create_from_execution_order(
        cls: ModalDependencies,
        execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    ) -> list[ModalDependencies]:
        modal_graph = defaultdict(lambda: {"previous": [], "next": []})
        in_degree, out_degree = defaultdict(int), defaultdict(int)
        for from_modal, to_modal in execution_order:
            modal_graph[from_modal]["next"].append(to_modal)
            modal_graph[to_modal]["previous"].append(from_modal)
            if from_modal not in in_degree:
                in_degree[from_modal] = 0
            if to_modal not in out_degree:
                out_degree[to_modal] = 0
            in_degree[to_modal] += 1
            out_degree[from_modal] += 1

        # Connect the last modals and the first modals
        zero_in_degree = [modal for modal, degree in in_degree.items() if degree == 0]
        zero_out_degree = [modal for modal, degree in out_degree.items() if degree == 0]
        for first_modal in zero_in_degree:
            for last_modal in zero_out_degree:
                modal_graph[first_modal]["previous"].append(last_modal)
                modal_graph[last_modal]["next"].append(first_modal)

        return [
            ModalDependencies(
                modal=modal,
                in_degree=in_degree[modal],
                out_degree=out_degree[modal],
                previous=dependencies["previous"],
                next=dependencies["next"],
            )
            for modal, dependencies in modal_graph.items()
        ]


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
        execution_order (list[tuple[PipelineTemplate, PipelineTemplate]]): The execution order of the modals.
            `MultiModalProcessGroupMesh` uses topological sort to determine the order of the modals.
            This is not related to actual execution order, but only used to assign ranks to modal models.
    """

    pp_axis, dp_axis, tp_axis = 0, 1, 2

    def __init__(
        self,
        modal_templates: dict[PipelineTemplate, int],
        execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert len(modal_templates) > 0, "At least one modal is required."
        for from_modal, to_modal in execution_order:
            assert (
                from_modal in modal_templates
            ), f"{from_modal} not in modal_templates."
            assert to_modal in modal_templates, f"{to_modal} not in modal_templates."

        self.modal_dependencies: list[ModalDependencies] = (
            ModalDependencies.create_from_execution_order(execution_order)
        )
        self.topological_sorted_modals = self.topological_sort(execution_order)
        assert len(modal_templates) == len(self.topological_sorted_modals)

        assert (
            dist.get_world_size()
            % sum(
                template.num_stages * tp_size
                for template, tp_size in modal_templates.items()
            )
            == 0
        ), "The world size must be divisible by tp_size * pp_size."
        dp_size = dist.get_world_size() // sum(
            template.num_stages * tp_size
            for template, tp_size in modal_templates.items()
        )

        max_tp_size = max(modal_templates.values())
        meshes: list[list[list[int]]] = []
        rank_index = 0
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)

        for modal in self.topological_sorted_modals:
            tp_size = modal_templates[modal]
            for _ in range(modal.num_stages):
                stage_mesh = []
                for _ in range(dp_size):
                    # create a list of ranks with length `max_tp_size`, where each rank is repeated `max_tp_size // tp_size` times.
                    # Example: [0, 0, 1, 1] for tp_size=2 and max_tp_size=4
                    ranks = [
                        i
                        for i in range(rank_index, rank_index + tp_size)
                        for _ in range(max_tp_size // tp_size)
                    ]
                    rank_index += tp_size

                    stage_mesh.append(ranks)
                    modal_to_ranks[modal].extend(ranks)
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
        self.stage_index_to_modal = list(
            itertools.chain.from_iterable(
                [modal] * modal.num_stages for modal in self.topological_sorted_modals
            )
        )

    @property
    def coords(self) -> list[tuple[int, ...]]:
        """The process coordinates.

        Returns:
            list[tuple[int, ...]]: The process coordinates.
        """
        return self._coords

    @property
    def mesh(self) -> dict[PipelineTemplate, np.ndarray]:
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

    def topological_sort(
        self, execution_order: list[tuple[PipelineTemplate, PipelineTemplate]]
    ) -> list[PipelineTemplate]:
        """
        Topological sort the modal templates based on the execution order.
        """
        graph: dict[PipelineTemplate, list[PipelineTemplate]] = defaultdict(list)
        in_degree: dict[PipelineTemplate, int] = defaultdict(int)
        out_degree: dict[PipelineTemplate, int] = defaultdict(int)

        for from_modal, to_modal in execution_order:
            graph[from_modal].append(to_modal)
            in_degree[to_modal] += 1
            out_degree[from_modal] += 1
            if from_modal not in in_degree:
                in_degree[from_modal] = 0
            if to_modal not in in_degree:
                in_degree[to_modal] = 0

        self.in_degree = copy.deepcopy(in_degree)
        self.out_degree = copy.deepcopy(out_degree)

        # Find all modals with no incoming edges
        zero_in_degree = deque([modal for modal in in_degree if in_degree[modal] == 0])
        topological_order: list[PipelineTemplate] = []

        while zero_in_degree:
            node = zero_in_degree.popleft()
            topological_order.append(node)

            # Reduce the in-degre of each neigher by 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        # Check if topological sort is possible
        if len(topological_order) == len(in_degree):
            return topological_order
        else:
            raise ValueError("Topological sort is not possible.")

    def create_group_along_axis(
        self,
        axis: int,
        indices_at_axis: list[int],
        backend: str | None = None,
    ) -> dist.ProcessGroup:
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        reduced_shape = list(self._shape)
        # the choices on the axis are reduced to 1, since it's determined by `indices_at_axis`
        reduced_shape[axis] = 1
        target_group = None
        # use Cartesian product to generate all combinations of coordinates
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(
                base_coord, axis, indices_at_axis
            )
            ranks_in_group = tuple(
                set([self._mesh[coord] for coord in coords_in_group])
            )

            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self,
        axis: int,
        indices_at_axis: list[int] | None = None,
        backend: str | None = None,
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
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))

        if axis == MultiModalProcessGroupMesh.pp_axis:
            coords_in_group = [
                self.get_coords_along_axis(coord, axis, indices_at_axis)
                for coord in self._coords
            ]
        else:
            coords_in_group = [
                self.get_coords_along_axis(self._coords[0], axis, indices_at_axis)
            ]

        ranks_in_group = [
            tuple(sorted(set([self._mesh[coord] for coord in coords])))
            for coords in coords_in_group
        ]

        process_group_list: list[dist.ProcessGroup] = []
        for ranks in ranks_in_group:
            if ranks not in self._ranks_to_group:
                group = self.create_group_along_axis(axis, indices_at_axis, backend)
                process_group_list.append(group)
            else:
                process_group_list.append(self._ranks_to_group[ranks])

        if len(process_group_list) == 1:
            return process_group_list[0]
        else:
            return process_group_list
