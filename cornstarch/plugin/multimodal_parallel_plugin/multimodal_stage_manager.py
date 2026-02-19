from __future__ import annotations

from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager

from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)


class MultiModalPipelineStageManager(PipelineStageManager):
    """PipelineStageManager for multimodal models with heterogeneous parallelism.

    Unlike a traditional unimodal pipeline where every stage follows the previous one
    linearly, multimodal pipelines execute encoder(s) and the LLM on disjoint sets of
    ranks that communicate only at modal boundaries.

    Key design differences from the parent class
    --------------------------------------------
    * Each modal has its own ``ProcessGroupMesh``; stages are numbered *within* the
      modal (0 … modal.num_stages-1), not globally.
    * ``get_prev_ranks`` / ``get_next_ranks`` return a list because a single rank may
      fan-out to (or receive from) multiple ranks at a modal boundary.
    * ``get_prev_rank`` / ``get_next_rank`` are intentionally removed.
    """

    def __init__(
        self,
        pg_mesh: MultiModalProcessGroupMesh,
        pipeline_axis: int,
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1

        # Convenience references
        my_modal = pg_mesh.my_modal
        my_mesh = pg_mesh.modal_meshes[my_modal]
        my_rank = dist.get_rank()
        my_coords = pg_mesh.coords  # list of (pp, dp, sp, tp) tuples, always len 1

        my_pp_stage = my_coords[0][pipeline_axis]
        my_modal_pp_size = my_mesh.shape[pipeline_axis]

        # ------------------------------------------------------------------
        # Compute prev/next ranks
        # ------------------------------------------------------------------
        prev_ranks_set: set[int] = set()
        next_ranks_set: set[int] = set()

        # Determine whether there are predecessor / successor modals
        has_prev_modal = bool(pg_mesh.backward_border_map) and any(
            my_rank in bmap
            for bmap in pg_mesh.backward_border_map.values()
        )
        has_next_modal = bool(pg_mesh.forward_border_map) and any(
            my_rank in fmap
            for fmap in pg_mesh.forward_border_map.values()
        )

        if my_pp_stage == 0 and has_prev_modal:
            # First stage in this modal: previous ranks come from the border map
            prev_ranks_set.update(pg_mesh.get_border_prev_ranks(my_rank))
        elif my_pp_stage > 0:
            # Not the first stage: predecessor is the previous PP row (same DP/SP/TP)
            pp_coord, dp_coord, sp_coord, tp_coord = my_coords[0]
            prev_pp = pp_coord - 1
            prev_rank = int(my_mesh[prev_pp, dp_coord, sp_coord, tp_coord])
            prev_ranks_set.add(prev_rank)

        if my_pp_stage == my_modal_pp_size - 1 and has_next_modal:
            # Last stage in this modal: next ranks come from the border map
            next_ranks_set.update(pg_mesh.get_border_next_ranks(my_rank))
        elif my_pp_stage < my_modal_pp_size - 1:
            # Not the last stage: successor is the next PP row (same DP/SP/TP)
            pp_coord, dp_coord, sp_coord, tp_coord = my_coords[0]
            next_pp = pp_coord + 1
            next_rank = int(my_mesh[next_pp, dp_coord, sp_coord, tp_coord])
            next_ranks_set.add(next_rank)

        self.prev_ranks: list[int] = sorted(prev_ranks_set)
        self.next_ranks: list[int] = sorted(next_ranks_set)

    # ------------------------------------------------------------------
    # Stage position helpers
    # ------------------------------------------------------------------

    @property
    def stage(self) -> int:
        """PP stage index *within the current modal* (0-based)."""
        return self.pg_mesh.coords[0][self.pipeline_axis]

    @property
    def num_stages(self) -> int:
        """Number of PP stages in the current modal."""
        return self.pg_mesh.modal_meshes[self.pg_mesh.my_modal].shape[
            self.pipeline_axis
        ]

    @property
    def num_stages_in_modal(self) -> int:
        """Alias for ``num_stages`` (stages are already modal-relative)."""
        return self.num_stages

    @property
    def stage_in_modal(self) -> int:
        """Stage index within the modal (same as ``stage``)."""
        return self.stage

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        """Return True if this rank is at the first PP stage.

        Args:
            check_only_in_modal: When True, check only whether this is the first
                stage *within* the current modal.  When False, additionally require
                that this modal is an encoder (or the LLM when no encoders exist),
                i.e. that there is no modal that feeds into it.
        """
        if self.stage != 0:
            return False
        if check_only_in_modal:
            return True
        # Global first: must be an encoder (or LLM when there are no encoders)
        my_modal = self.pg_mesh.my_modal
        if my_modal in self.pg_mesh.encoder_templates:
            return True
        if (
            not self.pg_mesh.encoder_templates
            and self.pg_mesh.llm_template is not None
            and my_modal == self.pg_mesh.llm_template[0]
        ):
            return True
        return False

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        """Return True if this rank is at the last PP stage.

        Args:
            check_only_in_modal: When True, check only whether this is the last
                stage *within* the current modal.  When False, additionally require
                that this modal is a decoder (or the LLM when there are no decoders).
        """
        if self.stage != self.num_stages - 1:
            return False
        if check_only_in_modal:
            return True
        # Global last: must be a decoder (or LLM when there are no decoders)
        my_modal = self.pg_mesh.my_modal
        if my_modal in self.pg_mesh.decoder_templates:
            return True
        if (
            not self.pg_mesh.decoder_templates
            and self.pg_mesh.llm_template is not None
            and my_modal == self.pg_mesh.llm_template[0]
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Rank accessors
    # ------------------------------------------------------------------

    def get_prev_rank(self) -> int:
        raise NotImplementedError(
            "get_prev_rank is removed from MultiModalPipelineStageManager. "
            "Use get_prev_ranks instead."
        )

    def get_next_rank(self) -> int:
        raise NotImplementedError(
            "get_next_rank is removed from MultiModalPipelineStageManager. "
            "Use get_next_ranks instead."
        )

    def get_prev_ranks(self) -> list[int]:
        return self.prev_ranks

    def get_next_ranks(self) -> list[int]:
        return self.next_ranks

    # ------------------------------------------------------------------
    # Process group helpers
    # ------------------------------------------------------------------

    def init_process_group_by_stages(
        self, stages: list[int]
    ) -> dist.ProcessGroup | list[dist.ProcessGroup]:
        """Get the intra-modal PP process group restricted to ``stages``."""
        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, stages)

    # ------------------------------------------------------------------
    # Layer distribution
    # ------------------------------------------------------------------

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        """Return layers-per-stage for the current modal.

        The returned list has length ``num_stages`` and its values sum to the
        total number of layers in the current modal.
        """
        return self.pg_mesh.my_modal.get_num_layers_per_stage()

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        """Return (start, end) layer indices for the given stage within the current modal.

        Args:
            layers_per_stage: Number of layers per stage (output of ``distribute_layers``).
            stage: Stage index within the current modal.  Defaults to ``self.stage``.

        Returns:
            (start_idx, end_idx) — exclusive end, relative to the modal's layer 0.
            Returns (0, 0) when the requested stage is outside this modal.
        """
        stage = self.stage if stage is None else stage
        if stage >= self.num_stages:
            return (0, 0)
        accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)
        return (int(accumulated[stage]), int(accumulated[stage + 1]))
