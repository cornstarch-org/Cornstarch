"""Tests for PipeweaverPipelineStageManager.

Verifies:
  - encoder_prev_ranks / encoder_next_ranks for all ranks
  - llm_prev_ranks / llm_next_ranks for all ranks
  - is_first_stage() / is_last_stage()
  - set_encoder_mode() / set_llm_mode() switches get_prev_ranks() / get_next_ranks()
  - stage property and distribute_layers / get_stage_index
"""
import pytest
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.pipeweaver_parallel_plugin.modal_process_group_mesh import (
    PipeweaverProcessGroupMesh,
)
from cornstarch.plugin.pipeweaver_parallel_plugin.pipeweaver_stage_manager import (
    PipeweaverPipelineStageManager,
)

from ..common import encoder1_template, encoder2_template, llm_template_2stages, llm_template_4stages


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


def _build(rank, world_size, encoder_template, llm_template, tp_size=1, sp_size=1):
    dist.init_process_group(
        backend="fake", store=FakeStore(), rank=rank, world_size=world_size
    )
    mesh = PipeweaverProcessGroupMesh(encoder_template, llm_template, tp_size, sp_size)
    sm = PipeweaverPipelineStageManager(mesh, mesh.pp_axis)
    return mesh, sm


# ---------------------------------------------------------------------------
# test_encoder_prev_next_ranks
#
# PP=2, TP=1, SP=1, DP=1 → world=2
#   rank 0: stage 0 → enc_prev=[], enc_next=[rank1]
#   rank 1: stage 1 → enc_prev=[rank0], enc_next=[rank0] (border: last→first)
# ---------------------------------------------------------------------------

def test_encoder_prev_next_ranks_pp2():
    for rank in range(2):
        mesh, sm = _build(rank, 2, encoder1_template, llm_template_2stages)
        if rank == 0:
            assert sm.encoder_prev_ranks == []
            assert sm.encoder_next_ranks == [1]
        else:  # rank 1 — last encoder stage → LLM first stage = rank 0
            assert sm.encoder_prev_ranks == [0]
            assert sm.encoder_next_ranks == [0]
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_llm_prev_next_ranks
#
# PP=2, TP=1, SP=1, DP=1 → world=2
#   rank 0: stage 0 → llm_prev=[rank1] (border), llm_next=[rank1]
#   rank 1: stage 1 → llm_prev=[rank0], llm_next=[]
# ---------------------------------------------------------------------------

def test_llm_prev_next_ranks_pp2():
    for rank in range(2):
        mesh, sm = _build(rank, 2, encoder1_template, llm_template_2stages)
        if rank == 0:
            assert sm.llm_prev_ranks == [1]  # border: LLM first ← encoder last
            assert sm.llm_next_ranks == [1]
        else:
            assert sm.llm_prev_ranks == [0]
            assert sm.llm_next_ranks == []
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_mode_switching
# ---------------------------------------------------------------------------

def test_mode_switching():
    for rank in range(2):
        mesh, sm = _build(rank, 2, encoder1_template, llm_template_2stages)

        # Default mode is encoder
        assert sm.current_mode == "encoder"
        assert sm.get_prev_ranks() == sm.encoder_prev_ranks
        assert sm.get_next_ranks() == sm.encoder_next_ranks

        sm.set_llm_mode()
        assert sm.current_mode == "llm"
        assert sm.get_prev_ranks() == sm.llm_prev_ranks
        assert sm.get_next_ranks() == sm.llm_next_ranks

        sm.set_encoder_mode()
        assert sm.current_mode == "encoder"
        assert sm.get_prev_ranks() == sm.encoder_prev_ranks
        assert sm.get_next_ranks() == sm.encoder_next_ranks

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_is_first_last_stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "encoder_template, llm_template, tp_size",
    [
        (encoder1_template, llm_template_2stages, 1),   # PP=2, world=2
        (llm_template_4stages, llm_template_4stages, 1),  # PP=4, world=4
        (encoder1_template, llm_template_2stages, 2),   # PP=2, TP=2, world=4
    ],
)
def test_is_first_last_stage(encoder_template, llm_template, tp_size):
    pp_size = encoder_template.num_stages
    world_size = pp_size * tp_size
    for rank in range(world_size):
        mesh, sm = _build(rank, world_size, encoder_template, llm_template, tp_size)
        pp_coord = mesh.coordinate(mesh.pp_axis)
        assert sm.is_first_stage() == (pp_coord == 0), (
            f"rank {rank}: is_first_stage mismatch (pp_coord={pp_coord})"
        )
        assert sm.is_last_stage() == (pp_coord == pp_size - 1), (
            f"rank {rank}: is_last_stage mismatch (pp_coord={pp_coord})"
        )
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_stage_property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "encoder_template, llm_template, tp_size",
    [
        (encoder1_template, llm_template_2stages, 1),
        (llm_template_4stages, llm_template_4stages, 2),
    ],
)
def test_stage_property(encoder_template, llm_template, tp_size):
    pp_size = encoder_template.num_stages
    world_size = pp_size * tp_size
    for rank in range(world_size):
        mesh, sm = _build(rank, world_size, encoder_template, llm_template, tp_size)
        expected_stage = mesh.coordinate(mesh.pp_axis)
        assert sm.stage == expected_stage, (
            f"rank {rank}: expected stage {expected_stage}, got {sm.stage}"
        )
        assert sm.num_stages == pp_size
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_encoder_next_ranks_multi_tp
#
# PP=2, TP=2, DP=1: ranks 0-3
#   shape (2, 1, 1, 2): (pp=0,dp=0,sp=0,tp=0)=rank0, (pp=0,dp=0,sp=0,tp=1)=rank1,
#                        (pp=1,dp=0,sp=0,tp=0)=rank2, (pp=1,dp=0,sp=0,tp=1)=rank3
#   Encoder mode:
#     rank 0 (pp=0): enc_next=[rank2]
#     rank 1 (pp=0): enc_next=[rank3]
#     rank 2 (pp=1): enc_next=[rank0]  ← border (last→first)
#     rank 3 (pp=1): enc_next=[rank1]  ← border
#   LLM mode:
#     rank 0 (pp=0): llm_prev=[rank2]  ← border
#     rank 1 (pp=0): llm_prev=[rank3]  ← border
#     rank 2 (pp=1): llm_prev=[rank0], llm_next=[]
#     rank 3 (pp=1): llm_prev=[rank1], llm_next=[]
# ---------------------------------------------------------------------------

def test_ranks_multi_tp():
    tp_size = 2
    pp_size = 2
    world_size = pp_size * tp_size  # 4
    expected_enc_next = {0: [2], 1: [3], 2: [0], 3: [1]}
    expected_enc_prev = {0: [], 1: [], 2: [0], 3: [1]}
    expected_llm_prev = {0: [2], 1: [3], 2: [0], 3: [1]}
    expected_llm_next = {0: [1], 1: [0], 2: [], 3: []}  # wait, let me recalculate

    # shape = (pp=2, dp=1, sp=1, tp=2)
    # ravel index in C order: rank = pp*(dp*sp*tp) + dp*(sp*tp) + sp*tp + tp_idx
    # rank 0: (0,0,0,0), rank 1: (0,0,0,1), rank 2: (1,0,0,0), rank 3: (1,0,0,1)
    # LLM next for rank 0 (pp=0): pp=1, same (d,s,t)=(0,0,0) → rank2
    # LLM next for rank 1 (pp=0): pp=1, same (d,s,t)=(0,0,1) → rank3
    # LLM prev for rank 2 (pp=1): pp=0, same (d,s,t)=(0,0,0) → rank0
    # LLM prev for rank 3 (pp=1): pp=0, same (d,s,t)=(0,0,1) → rank1

    expected_llm_next_correct = {0: [2], 1: [3], 2: [], 3: []}

    for rank in range(world_size):
        mesh, sm = _build(rank, world_size, encoder1_template, llm_template_2stages, tp_size)

        # Encoder mode
        assert sm.encoder_next_ranks == expected_enc_next[rank], (
            f"rank {rank}: encoder_next_ranks mismatch: "
            f"got {sm.encoder_next_ranks}, expected {expected_enc_next[rank]}"
        )
        assert sm.encoder_prev_ranks == expected_enc_prev[rank], (
            f"rank {rank}: encoder_prev_ranks mismatch"
        )

        # LLM mode
        assert sm.llm_prev_ranks == expected_llm_prev[rank], (
            f"rank {rank}: llm_prev_ranks mismatch: "
            f"got {sm.llm_prev_ranks}, expected {expected_llm_prev[rank]}"
        )
        assert sm.llm_next_ranks == expected_llm_next_correct[rank], (
            f"rank {rank}: llm_next_ranks mismatch: "
            f"got {sm.llm_next_ranks}, expected {expected_llm_next_correct[rank]}"
        )

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_distribute_layers_and_stage_index
# ---------------------------------------------------------------------------

def test_distribute_layers_and_stage_index():
    """PP=2, TP=1, DP=1: verify layer distribution for both modes."""
    world_size = 2
    for rank in range(world_size):
        mesh, sm = _build(rank, world_size, encoder1_template, llm_template_2stages)

        # Encoder mode
        sm.set_encoder_mode()
        enc_layers = sm.distribute_layers()
        assert enc_layers == encoder1_template.get_num_layers_per_stage(), (
            f"rank {rank}: encoder layers mismatch"
        )

        # LLM mode
        sm.set_llm_mode()
        llm_layers = sm.distribute_layers()
        assert llm_layers == llm_template_2stages.get_num_layers_per_stage(), (
            f"rank {rank}: llm layers mismatch"
        )

        # Stage index
        sm.set_encoder_mode()
        enc_layers = sm.distribute_layers()
        stage_idx = sm.get_stage_index(enc_layers)
        import numpy as np
        accumulated = np.insert(np.cumsum(enc_layers), 0, 0)
        expected = (int(accumulated[sm.stage]), int(accumulated[sm.stage + 1]))
        assert stage_idx == expected

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_larger_pp
#
# PP=4, TP=1, DP=1: world=4
#   Encoder: rank0→rank1→rank2→rank3→rank0 (border)
#   LLM:     rank4←rank3←rank2←rank1←rank0 (rank3 next=[], rank0 prev=[rank3])
# ---------------------------------------------------------------------------

def test_larger_pp():
    # Use llm_template_4stages for both (PP=4)
    pp_size = 4
    world_size = 4
    for rank in range(world_size):
        mesh, sm = _build(rank, world_size, llm_template_4stages, llm_template_4stages)

        if rank == 0:
            assert sm.encoder_prev_ranks == []
            assert sm.encoder_next_ranks == [1]
            assert sm.llm_prev_ranks == [3]  # border: from encoder last
            assert sm.llm_next_ranks == [1]
        elif rank == pp_size - 1:  # rank 3
            assert sm.encoder_prev_ranks == [rank - 1]
            assert sm.encoder_next_ranks == [0]  # border: to LLM first
            assert sm.llm_prev_ranks == [rank - 1]
            assert sm.llm_next_ranks == []
        else:
            assert sm.encoder_prev_ranks == [rank - 1]
            assert sm.encoder_next_ranks == [rank + 1]
            assert sm.llm_prev_ranks == [rank - 1]
            assert sm.llm_next_ranks == [rank + 1]

        dist.destroy_process_group()
