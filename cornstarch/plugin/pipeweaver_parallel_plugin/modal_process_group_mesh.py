from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh

from cornstarch.pipeline_template import PipelineTemplate


class PipeweaverProcessGroupMesh(ProcessGroupMesh):
    """Process group mesh for PipeWeaver co-located multimodal models.

    In PipeWeaver, encoder and LLM are partitioned into the same number of PP
    stages, and rank i co-hosts encoder stage i and LLM stage i. Both modals
    share the same DP / SP / TP / PP configuration.

    The mesh shape is ``[pp_size, dp_size, sp_size, tp_size]``.

    The encoder→LLM border wraps around: rank at (pp=N-1, d, s, t) sends to
    rank at (pp=0, d, s, t).  Gradients flow in reverse: (pp=0, d, s, t) →
    (pp=N-1, d, s, t).

    Args:
        encoder_template: PipelineTemplate for the encoder modal.
        llm_template: PipelineTemplate for the language model.
        tp_size: Shared tensor-parallel size for both modals.
        sp_size: Shared sequence-parallel size for both modals.
    """

    pp_axis, dp_axis, sp_axis, tp_axis = 0, 1, 2, 3

    def __init__(
        self,
        encoder_template: PipelineTemplate,
        llm_template: PipelineTemplate,
        tp_size: int = 1,
        sp_size: int = 1,
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert encoder_template.num_stages == llm_template.num_stages, (
            f"PipeWeaver requires encoder and LLM to have the same number of PP stages, "
            f"got encoder={encoder_template.num_stages}, llm={llm_template.num_stages}."
        )

        pp_size = encoder_template.num_stages
        num_ranks_per_replica = pp_size * sp_size * tp_size
        assert dist.get_world_size() % num_ranks_per_replica == 0, (
            f"World size {dist.get_world_size()} must be divisible by "
            f"ranks per replica {num_ranks_per_replica}."
        )
        dp_size = dist.get_world_size() // num_ranks_per_replica

        super().__init__(pp_size, dp_size, sp_size, tp_size)

        self.encoder_template = encoder_template
        self.llm_template = llm_template
        self._pp_size = pp_size
        self._dp_size = dp_size
        self._sp_size = sp_size
        self._tp_size = tp_size

        # Border maps: encoder last stage (pp=N-1) → LLM first stage (pp=0)
        # Both maps are 1:1 since TP/SP/DP are identical for both modals.
        self.encoder_to_llm_border_map: dict[int, int] = {}  # last-stage rank → first-stage rank
        self.llm_to_encoder_border_map: dict[int, int] = {}  # first-stage rank → last-stage rank

        for d in range(dp_size):
            for s in range(sp_size):
                for t in range(tp_size):
                    last_rank = int(ProcessGroupMesh.ravel(
                        (pp_size - 1, d, s, t), self._shape
                    ))
                    first_rank = int(ProcessGroupMesh.ravel(
                        (0, d, s, t), self._shape
                    ))
                    self.encoder_to_llm_border_map[last_rank] = first_rank
                    self.llm_to_encoder_border_map[first_rank] = last_rank

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def coords(self) -> tuple[int, ...]:
        """Coordinates of the current rank: (pp, dp, sp, tp)."""
        return self._coord

    # ------------------------------------------------------------------
    # Border helpers
    # ------------------------------------------------------------------

    def get_encoder_to_llm_next_rank(self, rank: int) -> Optional[int]:
        """Return the LLM-first-stage rank that ``rank`` (encoder last stage) sends to.

        Returns ``None`` if ``rank`` is not at the encoder's last PP stage.
        """
        return self.encoder_to_llm_border_map.get(rank, None)

    def get_llm_to_encoder_prev_rank(self, rank: int) -> Optional[int]:
        """Return the encoder-last-stage rank that sends to ``rank`` (LLM first stage).

        Returns ``None`` if ``rank`` is not at the LLM's first PP stage.
        """
        return self.llm_to_encoder_border_map.get(rank, None)

    # ------------------------------------------------------------------
    # Process group helpers
    # ------------------------------------------------------------------

    def get_pp_groups(self) -> list[dist.ProcessGroup]:
        """Return all PP process groups; the current rank belongs to exactly one."""
        pp_groups: list[dist.ProcessGroup] = []
        seen = set()
        pp_indices = list(range(self._pp_size))
        # Iterate over all (dp, sp, tp) combos to create every PP group.
        for d in range(self._dp_size):
            for s in range(self._sp_size):
                for t in range(self._tp_size):
                    ranks = tuple(
                        int(ProcessGroupMesh.ravel((pp, d, s, t), self._shape))
                        for pp in pp_indices
                    )
                    if ranks not in seen:
                        seen.add(ranks)
                        group = self._get_group(ranks)
                        if self._rank in ranks:
                            pp_groups.append(group)
        return pp_groups
