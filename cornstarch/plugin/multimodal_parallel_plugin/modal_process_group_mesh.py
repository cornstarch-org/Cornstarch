from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh

from cornstarch.pipeline_template import PipelineTemplate


class MultiModalProcessGroupMesh:
    """
    Process group mesh for multimodal models with heterogeneous parallel configurations.

    Instead of a single unified mesh with rank duplication, this class builds a separate
    mesh per modal. Adjacent modals are connected via explicit border rank maps that
    encode which ranks at the last PP stage of one modal correspond to which ranks at
    the first PP stage of the next modal.

    Each modal mesh has shape [pp_modal, dp, sp_modal, tp_modal]. Ranks are assigned
    sequentially (encoders first, then LLM), with no duplication across TP/SP slots.

    Example — Encoder TP=2, LLM TP=4, both PP=1, DP=1:
        Encoder mesh shape [1,1,1,2]:  ranks [[[[0, 1]]]]
        LLM     mesh shape [1,1,1,4]:  ranks [[[[2, 3, 4, 5]]]]
        forward_border_map[(encoder, llm)]:  {0: [2, 3], 1: [4, 5]}
        backward_border_map[(encoder, llm)]: {2: [0], 3: [0], 4: [1], 5: [1]}
    """

    pp_axis, dp_axis, sp_axis, tp_axis = 0, 1, 2, 3

    def __init__(
        self,
        encoder_templates: Optional[
            dict[PipelineTemplate, int | tuple[int, int]]
        ] = None,
        llm_template: Optional[tuple[PipelineTemplate, int, int]] = None,
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."

        if encoder_templates is None:
            encoder_templates = {}

        for encoder_template, parallel_sizes in list(encoder_templates.items()):
            if isinstance(parallel_sizes, int):
                encoder_templates[encoder_template] = (parallel_sizes, 1)

        self.encoder_templates = encoder_templates
        self.llm_template = llm_template
        self.decoder_templates: dict = {}
        self.topological_sorted_modals: list[PipelineTemplate] = [
            *encoder_templates.keys(),
            *([] if llm_template is None else [llm_template[0]]),
        ]

        # Compute number of ranks per model replica and DP size
        num_ranks_in_model = sum(
            template.num_stages * tp_size * sp_size
            for template, (tp_size, sp_size) in encoder_templates.items()
        ) + (
            0
            if llm_template is None
            else llm_template[0].num_stages * llm_template[1] * llm_template[2]
        )
        assert dist.get_world_size() % num_ranks_in_model == 0, (
            f"World size {dist.get_world_size()} must be divisible by "
            f"ranks per replica {num_ranks_in_model}."
        )
        dp_size = dist.get_world_size() // num_ranks_in_model
        self._dp_size = dp_size

        # Build ordered iterable: [(modal, tp_size, sp_size), ...]
        iterables: list[tuple[PipelineTemplate, int, int]] = [
            (modal, tp_size, sp_size)
            for modal, (tp_size, sp_size) in encoder_templates.items()
        ]
        if llm_template is not None:
            iterables.append(llm_template)

        # Build per-modal meshes of shape [pp, dp, sp, tp].
        # Ranks are assigned in C-order: pp outermost, tp innermost.
        modal_meshes: dict[PipelineTemplate, np.ndarray] = {}
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)
        rank_index = 0

        for modal, tp_size, sp_size in iterables:
            pp_size = modal.num_stages
            total = pp_size * dp_size * sp_size * tp_size
            mesh_arr = np.arange(rank_index, rank_index + total, dtype=int).reshape(
                pp_size, dp_size, sp_size, tp_size
            )
            rank_index += total
            modal_meshes[modal] = mesh_arr
            modal_to_ranks[modal].extend(mesh_arr.flatten().tolist())

        self.modal_meshes = modal_meshes
        self.modal_to_ranks: dict[PipelineTemplate, list[int]] = {
            modal: sorted(set(ranks)) for modal, ranks in modal_to_ranks.items()
        }

        self._rank = dist.get_rank()
        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}

        # Determine which modal this rank belongs to
        self.my_modal: PipelineTemplate = None
        for modal in self.topological_sorted_modals:
            if self._rank in self.modal_to_ranks[modal]:
                self.my_modal = modal
                break
        assert self.my_modal is not None, (
            f"Rank {self._rank} not found in any modal's rank set."
        )

        # Coordinates within my_modal's mesh; always exactly one entry since no duplication
        idx = np.where(modal_meshes[self.my_modal] == self._rank)
        self._coords: list[tuple[int, ...]] = [
            tuple(int(x) for x in coord) for coord in zip(*idx)
        ]

        # Compute forward and backward border maps between adjacent modals
        self.forward_border_map: dict[
            tuple[PipelineTemplate, PipelineTemplate], dict[int, list[int]]
        ] = {}
        self.backward_border_map: dict[
            tuple[PipelineTemplate, PipelineTemplate], dict[int, list[int]]
        ] = {}

        if llm_template is not None:
            for enc_template in encoder_templates.keys():
                self._build_border_maps(
                    enc_template,
                    modal_meshes[enc_template],
                    llm_template[0],
                    modal_meshes[llm_template[0]],
                    dp_size,
                )

    def _build_border_maps(
        self,
        modal_A: PipelineTemplate,
        A_mesh: np.ndarray,
        modal_B: PipelineTemplate,
        B_mesh: np.ndarray,
        dp_size: int,
    ) -> None:
        """Build forward and backward border rank maps between two adjacent modals.

        modal_A's last PP stage feeds into modal_B's first PP stage.

        For each rank at A's last stage, forward_border_map records which ranks at
        B's first stage it should send to.  The mapping is determined by TP and SP
        ratios: if tp_B > tp_A each A-TP rank fans out to (tp_B/tp_A) B-TP ranks;
        if tp_A > tp_B multiple A-TP ranks converge on each B-TP rank.
        """
        tp_A = A_mesh.shape[self.tp_axis]
        sp_A = A_mesh.shape[self.sp_axis]
        tp_B = B_mesh.shape[self.tp_axis]
        sp_B = B_mesh.shape[self.sp_axis]

        forward_map: dict[int, list[int]] = {}

        for d in range(dp_size):
            for s_A in range(sp_A):
                for t_A in range(tp_A):
                    a_rank = int(A_mesh[-1, d, s_A, t_A])
                    b_ranks: list[int] = []

                    # TP dimension mapping
                    if tp_B >= tp_A:
                        tp_ratio = tp_B // tp_A
                        t_B_range = range(t_A * tp_ratio, (t_A + 1) * tp_ratio)
                    else:
                        # fan-in: multiple A TP ranks share the same B TP rank
                        tp_ratio_inv = tp_A // tp_B
                        t_B_range = range(t_A // tp_ratio_inv, t_A // tp_ratio_inv + 1)

                    # SP dimension mapping: always include all LLM SP ranks.
                    # Unlike TP, the schedule cannot determine at this level which
                    # LLM SP rank needs which encoder SP rank's tokens (that is
                    # model-specific). Hooks registered on the schedule handle
                    # the actual per-rank routing.
                    s_B_range = range(0, sp_B)

                    for s_B in s_B_range:
                        for t_B in t_B_range:
                            b_ranks.append(int(B_mesh[0, d, s_B, t_B]))

                    forward_map[a_rank] = b_ranks

        # Derive backward map as the inverse of forward map
        backward_map: dict[int, list[int]] = defaultdict(list)
        for a_rank, b_ranks_list in forward_map.items():
            for b_rank in b_ranks_list:
                if a_rank not in backward_map[b_rank]:
                    backward_map[b_rank].append(a_rank)

        self.forward_border_map[(modal_A, modal_B)] = forward_map
        self.backward_border_map[(modal_A, modal_B)] = dict(backward_map)

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def coords(self) -> list[tuple[int, ...]]:
        """Coordinates of the current rank within its modal's mesh."""
        return self._coords

    @property
    def mesh(self) -> np.ndarray:
        """The current modal's process rank mesh."""
        return self.modal_meshes[self.my_modal]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the current modal's mesh."""
        return self.modal_meshes[self.my_modal].shape

    def size(self, axis: int) -> int:
        """Size along the given axis in the current modal's mesh."""
        return self.modal_meshes[self.my_modal].shape[axis]

    # ------------------------------------------------------------------
    # Border helpers
    # ------------------------------------------------------------------

    def get_border_next_ranks(self, rank: int) -> list[int]:
        """Return ranks in the next modal that ``rank`` should send to at a pipeline border.

        Encoder last-stage ranks return the LLM first-stage ranks they should send to.
        LLM last-stage ranks return an empty list (no downstream modal).
        """
        result: list[int] = []
        for fmap in self.forward_border_map.values():
            if rank in fmap:
                result.extend(fmap[rank])
        return sorted(result)

    def get_border_prev_ranks(self, rank: int) -> list[int]:
        """Return ranks in the previous modal(s) that send to ``rank``.

        Accumulates across all backward border maps so that, for example, a LLM
        first-stage rank correctly returns last-stage ranks from *every* encoder
        when there are multiple encoders.
        """
        result: list[int] = []
        for bmap in self.backward_border_map.values():
            if rank in bmap:
                result.extend(bmap[rank])
        return sorted(result)

    # ------------------------------------------------------------------
    # Process group creation
    # ------------------------------------------------------------------

    def _get_group(
        self, ranks: tuple[int, ...], backend: Optional[str] = None
    ) -> dist.ProcessGroup:
        if ranks not in self._ranks_to_group:
            group = dist.new_group(ranks=list(ranks), backend=backend)
            self._ranks_to_group[ranks] = group
            self._group_to_ranks[group] = ranks
        return self._ranks_to_group[ranks]

    @staticmethod
    def unravel(rank: int, mesh: np.ndarray) -> list[tuple[int, ...]]:
        """Convert a rank to a list of coordinates within a mesh.

        In the new per-modal design each rank appears at exactly one coordinate,
        so this always returns a single-element list.  The method is kept for
        backward compatibility with subclasses.
        """
        indices = np.where(mesh == rank)
        return list(zip(*indices))

    def create_or_get_group_along_axis(
        self,
        axis: int | list[int],
        indices_at_axis: list[int] | list[list[int]],
        target_ranks_in_group: tuple[int, ...],
        backend: Optional[str] = None,
        modal: Optional[PipelineTemplate] = None,
    ) -> dist.ProcessGroup:
        """Create or retrieve the process group that exactly matches ``target_ranks_in_group``.

        All processes across all modals must call this method collectively so that
        ``dist.new_group`` is invoked consistently.  The ``modal`` parameter pins
        the search to a specific modal's mesh; defaults to the calling rank's modal.
        """
        modal = modal or self.my_modal

        if isinstance(axis, int):
            axis = [axis]
            if isinstance(indices_at_axis[0], int):
                indices_at_axis = [indices_at_axis]

        mesh = self.modal_meshes[modal]
        shape = list(mesh.shape)
        indices_at_axis = indices_at_axis or [list(range(shape[ax])) for ax in axis]
        reduced_shape = shape[:]
        for ax in axis:
            reduced_shape[ax] = 1

        target_group = None
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(
                base_coord, axis, indices_at_axis
            )
            ranks_in_group = tuple(
                sorted(set(int(mesh[coord]) for coord in coords_in_group))
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
        """Get the process group(s) along the given axis for the current rank.

        All processes must call this method collectively (required by
        ``dist.new_group``).  Internally, groups are created for every modal so
        that all ranks participate.

        Args:
            axis: Axis (or list of axes) along which to form the group.
            indices_at_axis: Specific indices to include.  Defaults to all indices.
            backend: Distributed backend.  Defaults to None.

        Returns:
            A single ``dist.ProcessGroup`` for non-PP axes, or a
            ``list[dist.ProcessGroup]`` when the PP axis is requested.
        """
        if isinstance(axis, int):
            axis_list = [axis]
            if indices_at_axis is not None and not isinstance(indices_at_axis[0], list):
                indices_at_axis = [indices_at_axis]
        else:
            axis_list = list(axis)

        is_pp = self.pp_axis in axis_list

        my_groups: list[dist.ProcessGroup] = []

        # Iterate over ALL modals so every rank participates in new_group calls.
        for modal, mesh in self.modal_meshes.items():
            shape = list(mesh.shape)
            _indices = indices_at_axis if indices_at_axis is not None else [
                list(range(shape[ax])) for ax in axis_list
            ]
            reduced_shape = shape[:]
            for ax in axis_list:
                reduced_shape[ax] = 1

            for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
                coords_in_group = ProcessGroupMesh.get_coords_along_axis(
                    base_coord, axis_list, _indices
                )
                ranks_in_group = tuple(
                    sorted(set(int(mesh[coord]) for coord in coords_in_group))
                )
                group = self._get_group(ranks_in_group, backend=backend)
                if self._rank in ranks_in_group and modal == self.my_modal:
                    my_groups.append(group)

        # Deduplicate while preserving order
        my_groups = list(dict.fromkeys(my_groups))

        if is_pp or len(my_groups) > 1:
            return my_groups
        elif my_groups:
            return my_groups[0]
        return None

    def get_global_pp_group(self) -> dist.ProcessGroup:
        """Return a process group spanning all pipeline stages of one DP replica.

        This group covers all modals (encoder + LLM) that are part of the same
        data-parallel replica and is used by the optimizer for gradient-norm
        computation across all pipeline stages.

        All ranks must call this method collectively.
        """
        dp_size = self._dp_size
        my_dp_idx = self._coords[0][self.dp_axis]

        target_group = None
        for d in range(dp_size):
            ranks_in_replica: list[int] = []
            for mesh in self.modal_meshes.values():
                for pp in range(mesh.shape[self.pp_axis]):
                    for s in range(mesh.shape[self.sp_axis]):
                        for t in range(mesh.shape[self.tp_axis]):
                            ranks_in_replica.append(int(mesh[pp, d, s, t]))
            ranks_tuple = tuple(sorted(set(ranks_in_replica)))
            group = self._get_group(ranks_tuple)
            if d == my_dp_idx:
                target_group = group

        return target_group
