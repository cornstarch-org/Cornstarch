from __future__ import annotations

from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from colossalai.pipeline.stage_manager import PipelineStageManager

from cornstarch.plugin.pipeweaver_parallel_plugin.modal_process_group_mesh import (
    PipeweaverProcessGroupMesh,
)


class PipeweaverPipelineStageManager(PipelineStageManager):
    """PipelineStageManager for PipeWeaver co-located multimodal pipelines.

    In PipeWeaver, each rank hosts both an encoder stage and an LLM stage at
    the same PP index.  The stage manager has two communication modes:

    * **encoder mode** — prev/next ranks follow the encoder pipeline direction.
      The last encoder stage (rank N-1) wraps around to rank 0 (LLM first).
    * **llm mode**     — prev/next ranks follow the LLM pipeline direction.
      The first LLM stage (rank 0) wraps around from rank N-1 (encoder last).

    Use :meth:`set_encoder_mode` / :meth:`set_llm_mode` before P2P calls.
    """

    def __init__(
        self,
        pg_mesh: PipeweaverProcessGroupMesh,
        pipeline_axis: int,
    ) -> None:
        assert isinstance(pg_mesh, PipeweaverProcessGroupMesh)
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1
        self.use_zbv = False

        my_rank = dist.get_rank()
        coords = pg_mesh.coords  # (pp, dp, sp, tp)
        pp_coord, dp_coord, sp_coord, tp_coord = coords
        num_stages = pg_mesh.size(pipeline_axis)
        mesh_shape = pg_mesh.shape

        # ----------------------------------------------------------------
        # Encoder mode prev/next ranks
        # ----------------------------------------------------------------
        if pp_coord == 0:
            self.encoder_prev_ranks: list[int] = []
        else:
            prev_rank = int(
                ProcessGroupMesh.ravel(
                    (pp_coord - 1, dp_coord, sp_coord, tp_coord), mesh_shape
                )
            )
            self.encoder_prev_ranks = [prev_rank]

        if pp_coord == num_stages - 1:
            # Last encoder stage: wraps to LLM first stage (rank 0 slice)
            llm_first_rank = pg_mesh.get_encoder_to_llm_next_rank(my_rank)
            self.encoder_next_ranks: list[int] = [llm_first_rank]
        else:
            next_rank = int(
                ProcessGroupMesh.ravel(
                    (pp_coord + 1, dp_coord, sp_coord, tp_coord), mesh_shape
                )
            )
            self.encoder_next_ranks = [next_rank]

        # ----------------------------------------------------------------
        # LLM mode prev/next ranks
        # ----------------------------------------------------------------
        if pp_coord == 0:
            # First LLM stage: wraps to encoder last stage (rank N-1 slice)
            enc_last_rank = pg_mesh.get_llm_to_encoder_prev_rank(my_rank)
            self.llm_prev_ranks: list[int] = [enc_last_rank]
        else:
            prev_rank = int(
                ProcessGroupMesh.ravel(
                    (pp_coord - 1, dp_coord, sp_coord, tp_coord), mesh_shape
                )
            )
            self.llm_prev_ranks = [prev_rank]

        if pp_coord == num_stages - 1:
            self.llm_next_ranks: list[int] = []
        else:
            next_rank = int(
                ProcessGroupMesh.ravel(
                    (pp_coord + 1, dp_coord, sp_coord, tp_coord), mesh_shape
                )
            )
            self.llm_next_ranks = [next_rank]

        # Default mode is encoder
        self._mode: str = "encoder"

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_encoder_mode(self) -> None:
        """Switch to encoder communication mode."""
        self._mode = "encoder"

    def set_llm_mode(self) -> None:
        """Switch to LLM communication mode."""
        self._mode = "llm"

    @property
    def current_mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Stage position helpers
    # ------------------------------------------------------------------

    @property
    def stage(self) -> int:
        """PP stage index (0-based), shared by both encoder and LLM."""
        return self.pg_mesh.coords[self.pipeline_axis]

    @property
    def num_stages(self) -> int:
        """Total number of PP stages."""
        return self.pg_mesh.size(self.pipeline_axis)

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == 0

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == self.num_stages - 1

    # ------------------------------------------------------------------
    # Rank accessors (mode-aware)
    # ------------------------------------------------------------------

    def get_prev_rank(self) -> int:
        raise NotImplementedError(
            "get_prev_rank is removed from PipeweaverPipelineStageManager. "
            "Use get_prev_ranks instead."
        )

    def get_next_rank(self) -> int:
        raise NotImplementedError(
            "get_next_rank is removed from PipeweaverPipelineStageManager. "
            "Use get_next_ranks instead."
        )

    def get_prev_ranks(self) -> list[int]:
        """Return prev ranks for the current mode (encoder or llm)."""
        if self._mode == "encoder":
            return self.encoder_prev_ranks
        return self.llm_prev_ranks

    def get_next_ranks(self) -> list[int]:
        """Return next ranks for the current mode (encoder or llm)."""
        if self._mode == "encoder":
            return self.encoder_next_ranks
        return self.llm_next_ranks

    # ------------------------------------------------------------------
    # Process group helpers
    # ------------------------------------------------------------------

    def init_process_group_by_stages(
        self, stages: list[int]
    ) -> dist.ProcessGroup | list[dist.ProcessGroup]:
        """Get the PP process group restricted to the given stage indices."""
        return self.pg_mesh.create_group_along_axis(self.pipeline_axis, stages)

    # ------------------------------------------------------------------
    # Layer distribution
    # ------------------------------------------------------------------

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        """Return layers-per-stage for the current modal (encoder or LLM).

        In encoder mode returns encoder template layers; in LLM mode returns
        LLM template layers.
        """
        if self._mode == "encoder":
            return self.pg_mesh.encoder_template.get_num_layers_per_stage()
        return self.pg_mesh.llm_template.get_num_layers_per_stage()

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        """Return (start, end) layer indices for ``stage`` within the current modal."""
        stage = self.stage if stage is None else stage
        if stage >= len(layers_per_stage):
            return (0, 0)
        accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)
        return (int(accumulated[stage]), int(accumulated[stage + 1]))
