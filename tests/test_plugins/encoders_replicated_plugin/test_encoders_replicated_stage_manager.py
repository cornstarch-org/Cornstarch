import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.encoders_replicated_plugin.encoders_replicated_stage_manager import (
    EncodersReplicatedPipelineStageManager,
)
from cornstarch.plugin.encoders_replicated_plugin.process_group_mesh import (
    EncodersReplicatedProcessGroupMesh,
)

from ..common import (
    encoder1_template,
    encoder2_template,
    encoder3_template,
    llm_template_2stages,
    llm_template_4stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "llm_template, expected_prev_next_ranks",
    [
        (
            (llm_template_2stages, 1, 1),
            [
                {"prev": [2], "next": [2]},
                {"prev": [3], "next": [3]},
                {"prev": [0], "next": [0]},
                {"prev": [1], "next": [1]},
            ],
        ),
        (
            (llm_template_2stages, 2, 2),
            [
                {"prev": [8], "next": [8]},
                {"prev": [9], "next": [9]},
                {"prev": [10], "next": [10]},
                {"prev": [11], "next": [11]},
                {"prev": [12], "next": [12]},
                {"prev": [13], "next": [13]},
                {"prev": [14], "next": [14]},
                {"prev": [15], "next": [15]},
                {"prev": [0], "next": [0]},
                {"prev": [1], "next": [1]},
                {"prev": [2], "next": [2]},
                {"prev": [3], "next": [3]},
                {"prev": [4], "next": [4]},
                {"prev": [5], "next": [5]},
                {"prev": [6], "next": [6]},
                {"prev": [7], "next": [7]},
            ],
        ),
        (
            (llm_template_4stages, 4, 1),
            [
                {"prev": [24], "next": [8]},
                {"prev": [25], "next": [9]},
                {"prev": [26], "next": [10]},
                {"prev": [27], "next": [11]},
                {"prev": [28], "next": [12]},
                {"prev": [29], "next": [13]},
                {"prev": [30], "next": [14]},
                {"prev": [31], "next": [15]},
                {"prev": [0], "next": [16]},
                {"prev": [1], "next": [17]},
                {"prev": [2], "next": [18]},
                {"prev": [3], "next": [19]},
                {"prev": [4], "next": [20]},
                {"prev": [5], "next": [21]},
                {"prev": [6], "next": [22]},
                {"prev": [7], "next": [23]},
                {"prev": [8], "next": [24]},
                {"prev": [9], "next": [25]},
                {"prev": [10], "next": [26]},
                {"prev": [11], "next": [27]},
                {"prev": [12], "next": [28]},
                {"prev": [13], "next": [29]},
                {"prev": [14], "next": [30]},
                {"prev": [15], "next": [31]},
                {"prev": [16], "next": [0]},
                {"prev": [17], "next": [1]},
                {"prev": [18], "next": [2]},
                {"prev": [19], "next": [3]},
                {"prev": [20], "next": [4]},
                {"prev": [21], "next": [5]},
                {"prev": [22], "next": [6]},
                {"prev": [23], "next": [7]},
            ],
        ),
        (
            (llm_template_4stages, 1, 4),
            [
                {"prev": [24], "next": [8]},
                {"prev": [25], "next": [9]},
                {"prev": [26], "next": [10]},
                {"prev": [27], "next": [11]},
                {"prev": [28], "next": [12]},
                {"prev": [29], "next": [13]},
                {"prev": [30], "next": [14]},
                {"prev": [31], "next": [15]},
                {"prev": [0], "next": [16]},
                {"prev": [1], "next": [17]},
                {"prev": [2], "next": [18]},
                {"prev": [3], "next": [19]},
                {"prev": [4], "next": [20]},
                {"prev": [5], "next": [21]},
                {"prev": [6], "next": [22]},
                {"prev": [7], "next": [23]},
                {"prev": [8], "next": [24]},
                {"prev": [9], "next": [25]},
                {"prev": [10], "next": [26]},
                {"prev": [11], "next": [27]},
                {"prev": [12], "next": [28]},
                {"prev": [13], "next": [29]},
                {"prev": [14], "next": [30]},
                {"prev": [15], "next": [31]},
                {"prev": [16], "next": [0]},
                {"prev": [17], "next": [1]},
                {"prev": [18], "next": [2]},
                {"prev": [19], "next": [3]},
                {"prev": [20], "next": [4]},
                {"prev": [21], "next": [5]},
                {"prev": [22], "next": [6]},
                {"prev": [23], "next": [7]},
            ],
        ),
    ],
    ids=["pp2_tp1_sp1", "pp2_tp2_sp2", "pp4_tp4_sp1", "pp4_tp1_sp4"],
)
@pytest.mark.parametrize(
    "encoder_templates",
    [
        ([encoder1_template]),
        ([encoder3_template]),
        ([encoder1_template, encoder3_template]),
    ],
    ids=["enc0", "enc1", "enc2"],
)
def test_encoders_replicated_pipeline_stage_manager(
    encoder_templates: list[PipelineTemplate],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    tp_size, sp_size = llm_template[1], llm_template[2]
    world_size = 2 * llm_template[0].num_stages * tp_size * sp_size
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = EncodersReplicatedProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = EncodersReplicatedPipelineStageManager(mesh, mesh.pp_axis)

        assert stage_manager.prev_ranks == expected_prev_next_ranks[rank]["prev"], (
            f"rank {rank} expected to have {expected_prev_next_ranks[rank]['prev']} as previous ranks, "
            f"but got {stage_manager.prev_ranks}."
        )
        assert stage_manager.next_ranks == expected_prev_next_ranks[rank]["next"], (
            f"rank {rank} expected to have {expected_prev_next_ranks[rank]['next']} as next ranks, "
            f"but got {stage_manager.next_ranks}."
        )

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "llm_template, expected_first_last_stages",
    [
        (
            (llm_template_2stages, 1, 1),
            {
                (0, 1): (True, False),
                (2, 3): (False, True),
            },
        ),
        (
            (llm_template_2stages, 2, 2),
            {
                tuple(range(0, 8)): (True, False),
                tuple(range(8, 16)): (False, True),
            },
        ),
        (
            (llm_template_4stages, 4, 1),
            {
                tuple(range(0, 8)): (True, False),
                tuple(range(8, 24)): (False, False),
                tuple(range(24, 32)): (False, True),
            },
        ),
        (
            (llm_template_4stages, 1, 4),
            {
                tuple(range(0, 8)): (True, False),
                tuple(range(8, 24)): (False, False),
                tuple(range(24, 32)): (False, True),
            },
        ),
    ],
    ids=["pp2_tp1_sp1", "pp2_tp2_sp2", "pp4_tp4_sp1", "pp4_tp1_sp4"],
)
def test_first_last_stage(
    llm_template: tuple[PipelineTemplate, int, int],
    expected_first_last_stages: dict[tuple[int], tuple[bool, bool]],
):
    tp_size, sp_size = llm_template[1], llm_template[2]
    world_size = 2 * llm_template[0].num_stages * tp_size * sp_size
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = EncodersReplicatedProcessGroupMesh(llm_template)
        stage_manager = EncodersReplicatedPipelineStageManager(mesh, mesh.pp_axis)

        expected_first_last_stage = next(
            value
            for ranks, value in expected_first_last_stages.items()
            if rank in ranks
        )
        assert expected_first_last_stage == (
            stage_manager.is_first_stage(check_only_in_modal=False),
            stage_manager.is_last_stage(check_only_in_modal=False),
        ), (
            f"rank {rank} expected to have {expected_first_last_stage} as first and last stage, "
            f"but got ({stage_manager.is_first_stage(check_only_in_modal=False), stage_manager.is_last_stage(check_only_in_modal=False)})."
        )

        # check automatic behavior, which should be the same with check_only_in_modal=True
        assert expected_first_last_stage == (
            stage_manager.is_first_stage(),
            stage_manager.is_last_stage(),
        ), (
            f"rank {rank} expected to have {expected_first_last_stage} as first and last stage in modal, "
            f"but got ({stage_manager.is_first_stage(), stage_manager.is_last_stage()})."
        )

        dist.destroy_process_group()
