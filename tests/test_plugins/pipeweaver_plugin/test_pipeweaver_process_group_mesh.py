"""Tests for PipeweaverProcessGroupMesh.

Verifies:
  - Single unified mesh shape and rank assignments
  - encoder_to_llm_border_map: each rank at pp=N-1 maps to the corresponding
    rank at pp=0 with the same (dp, sp, tp) coordinates.
  - llm_to_encoder_border_map: inverse of the above.
  - get_encoder_to_llm_next_rank / get_llm_to_encoder_prev_rank accessors.
"""
import pytest
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.pipeweaver_parallel_plugin.modal_process_group_mesh import (
    PipeweaverProcessGroupMesh,
)

from ..common import encoder1_template, encoder2_template, llm_template_2stages, llm_template_4stages


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_mesh(
    rank: int,
    world_size: int,
    encoder_template: PipelineTemplate,
    llm_template: PipelineTemplate,
    tp_size: int = 1,
    sp_size: int = 1,
) -> PipeweaverProcessGroupMesh:
    dist.init_process_group(
        backend="fake", store=FakeStore(), rank=rank, world_size=world_size
    )
    return PipeweaverProcessGroupMesh(
        encoder_template=encoder_template,
        llm_template=llm_template,
        tp_size=tp_size,
        sp_size=sp_size,
    )


# ---------------------------------------------------------------------------
# test_mesh_shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "encoder_template, llm_template, tp_size, sp_size, expected_shape",
    [
        # PP=2, TP=1, SP=1, DP=1 → world=2
        (encoder1_template, llm_template_2stages, 1, 1, (2, 1, 1, 1)),
        # PP=2, TP=2, SP=1, DP=1 → world=4
        (encoder1_template, llm_template_2stages, 2, 1, (2, 1, 1, 2)),
        # PP=2, TP=1, SP=1, DP=2 → world=4
        (encoder1_template, llm_template_2stages, 1, 1, (2, 2, 1, 1)),
        # PP=4, TP=2, SP=1, DP=1 → world=8
        (llm_template_4stages, llm_template_4stages, 2, 1, (4, 1, 1, 2)),
    ],
)
def test_mesh_shape(
    encoder_template,
    llm_template,
    tp_size,
    sp_size,
    expected_shape,
):
    pp_size = encoder_template.num_stages
    world_size = pp_size * expected_shape[1] * sp_size * tp_size
    for rank in range(world_size):
        mesh = _build_mesh(rank, world_size, encoder_template, llm_template, tp_size, sp_size)
        assert mesh.shape == expected_shape, (
            f"rank {rank}: expected shape {expected_shape}, got {mesh.shape}"
        )
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_border_maps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "encoder_template, llm_template, tp_size, sp_size",
    [
        (encoder1_template, llm_template_2stages, 1, 1),  # PP=2, world=2
        (encoder1_template, llm_template_2stages, 2, 1),  # PP=2, TP=2, world=4
        (encoder1_template, llm_template_2stages, 1, 1),  # PP=2, DP=2, world=4
        (llm_template_4stages, llm_template_4stages, 2, 1),  # PP=4, TP=2, world=8
    ],
)
def test_border_maps(encoder_template, llm_template, tp_size, sp_size):
    pp_size = encoder_template.num_stages
    # Calculate dp_size to fill world_size
    # We test with dp=1 here; dp is inferred from world_size / (pp*tp*sp)
    world_size = pp_size * tp_size * sp_size  # dp=1
    for rank in range(world_size):
        mesh = _build_mesh(rank, world_size, encoder_template, llm_template, tp_size, sp_size)
        shape = mesh.shape  # (pp, dp, sp, tp)

        # Build expected border map manually
        expected_enc_to_llm: dict[int, int] = {}
        expected_llm_to_enc: dict[int, int] = {}
        dp_size = shape[1]
        for d in range(dp_size):
            for s in range(sp_size):
                for t in range(tp_size):
                    last = int(ProcessGroupMesh.ravel((pp_size - 1, d, s, t), shape))
                    first = int(ProcessGroupMesh.ravel((0, d, s, t), shape))
                    expected_enc_to_llm[last] = first
                    expected_llm_to_enc[first] = last

        assert mesh.encoder_to_llm_border_map == expected_enc_to_llm, (
            f"rank {rank}: encoder_to_llm_border_map mismatch"
        )
        assert mesh.llm_to_encoder_border_map == expected_llm_to_enc, (
            f"rank {rank}: llm_to_encoder_border_map mismatch"
        )

        # Check accessor methods
        pp_coord = ProcessGroupMesh.unravel(rank, shape)[0]
        if pp_coord == pp_size - 1:
            assert mesh.get_encoder_to_llm_next_rank(rank) == expected_enc_to_llm[rank], (
                f"rank {rank}: get_encoder_to_llm_next_rank mismatch"
            )
        else:
            assert mesh.get_encoder_to_llm_next_rank(rank) is None

        if pp_coord == 0:
            assert mesh.get_llm_to_encoder_prev_rank(rank) == expected_llm_to_enc[rank], (
                f"rank {rank}: get_llm_to_encoder_prev_rank mismatch"
            )
        else:
            assert mesh.get_llm_to_encoder_prev_rank(rank) is None

        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# test_border_maps_with_dp
# ---------------------------------------------------------------------------

def test_border_maps_with_dp():
    """PP=2, TP=2, DP=2 (world=8): verify 1:1 border mapping per dp slice."""
    encoder_template = encoder1_template  # 2 stages
    llm_template = llm_template_2stages   # 2 stages
    tp_size = 2
    sp_size = 1
    pp_size = 2
    dp_size = 2
    world_size = pp_size * dp_size * sp_size * tp_size  # 8

    for rank in range(world_size):
        mesh = _build_mesh(rank, world_size, encoder_template, llm_template, tp_size, sp_size)
        shape = mesh.shape  # (2, 2, 1, 2)

        for d in range(dp_size):
            for t in range(tp_size):
                last = int(ProcessGroupMesh.ravel((pp_size - 1, d, sp_size - 1, t), shape))
                first = int(ProcessGroupMesh.ravel((0, d, sp_size - 1, t), shape))
                assert mesh.encoder_to_llm_border_map[last] == first
                assert mesh.llm_to_encoder_border_map[first] == last

        dist.destroy_process_group()
