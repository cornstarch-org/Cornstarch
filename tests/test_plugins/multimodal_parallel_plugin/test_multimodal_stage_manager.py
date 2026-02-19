import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)

from ..common import (
    encoder1_template,
    encoder2_template,
    llm_template_2stages,
    llm_template_4stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_prev_next_ranks",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            [
                {"prev": [], "next": [4]},
                {"prev": [], "next": [5]},
                {"prev": [], "next": [6]},
                {"prev": [], "next": [7]},
                {"prev": [0], "next": [8, 9]},  # rank = 4
                {"prev": [1], "next": [10, 11]},
                {"prev": [2], "next": [12, 13]},
                {"prev": [3], "next": [14, 15]},
                {"prev": [4], "next": [16]},  # rank = 8
                {"prev": [4], "next": [17]},
                {"prev": [5], "next": [18]},
                {"prev": [5], "next": [19]},
                {"prev": [6], "next": [20]},  # rank = 12
                {"prev": [6], "next": [21]},
                {"prev": [7], "next": [22]},
                {"prev": [7], "next": [23]},
                {"prev": [8], "next": []},  # rank = 16
                {"prev": [9], "next": []},
                {"prev": [10], "next": []},
                {"prev": [11], "next": []},
                {"prev": [12], "next": []},  # rank = 20
                {"prev": [13], "next": []},
                {"prev": [14], "next": []},
                {"prev": [15], "next": []},
            ],
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            [
                {"prev": [], "next": [2]},  # rank = 0. encoder1 stage 0
                {"prev": [], "next": [3]},
                {"prev": [0], "next": [10, 11]},  # rank = 2. encoder1 stage 1 → llm
                {"prev": [1], "next": [12, 13]},
                {"prev": [], "next": [6]},  # rank = 4. encoder2 stage 0
                {"prev": [], "next": [7]},
                {"prev": [4], "next": [8]},
                {"prev": [5], "next": [9]},
                {"prev": [6], "next": [10, 11]},  # rank = 8. encoder2 last → llm
                {"prev": [7], "next": [12, 13]},
                {"prev": [2, 8], "next": [14]},
                {"prev": [2, 8], "next": [15]},
                {"prev": [3, 9], "next": [16]},  # rank = 12
                {"prev": [3, 9], "next": [17]},
                {"prev": [10], "next": []},  # rank = 14. LLM last stage
                {"prev": [11], "next": []},
                {"prev": [12], "next": []},  # rank = 16
                {"prev": [13], "next": []},
            ],
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            # encoder2 stage 0: ranks 0-11, no prev modal
            [{"prev": [], "next": [i + 12]} for i in range(0, 12)]
            # encoder2 stages 1-2 and LLM stages 0-2: intra-modal PP
            + [{"prev": [i - 12], "next": [i + 12]} for i in range(12, 72)]
            # LLM stage 3 (last): ranks 72-83, no next modal
            + [{"prev": [i - 12], "next": []} for i in range(72, 84)],
        ),
        (
            40,  # 20 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            [
                {"prev": [], "next": [4]},  # rank = 0. encoder stage 0
                {"prev": [], "next": [5]},
                {"prev": [], "next": [6]},
                {"prev": [], "next": [7]},
                {"prev": [0], "next": [8, 9, 12, 13]},  # rank = 4. encoder stage 1 → llm
                {"prev": [1], "next": [10, 11, 14, 15]},
                {"prev": [2], "next": [16, 17, 20, 21]},
                {"prev": [3], "next": [18, 19, 22, 23]},
                {"prev": [4], "next": [24]},  # rank = 8. LLM stage 0
                {"prev": [4], "next": [25]},
                {"prev": [5], "next": [26]},
                {"prev": [5], "next": [27]},
                {"prev": [4], "next": [28]},  # rank = 12
                {"prev": [4], "next": [29]},
                {"prev": [5], "next": [30]},
                {"prev": [5], "next": [31]},
                {"prev": [6], "next": [32]},  # rank = 16
                {"prev": [6], "next": [33]},
                {"prev": [7], "next": [34]},
                {"prev": [7], "next": [35]},
                {"prev": [6], "next": [36]},  # rank = 20
                {"prev": [6], "next": [37]},
                {"prev": [7], "next": [38]},
                {"prev": [7], "next": [39]},
                {"prev": [8], "next": []},  # rank = 24. LLM last stage
                {"prev": [9], "next": []},
                {"prev": [10], "next": []},
                {"prev": [11], "next": []},
                {"prev": [12], "next": []},  # rank = 28
                {"prev": [13], "next": []},
                {"prev": [14], "next": []},
                {"prev": [15], "next": []},
                {"prev": [16], "next": []},  # rank = 32
                {"prev": [17], "next": []},
                {"prev": [18], "next": []},
                {"prev": [19], "next": []},
                {"prev": [20], "next": []},  # rank = 36
                {"prev": [21], "next": []},
                {"prev": [22], "next": []},
                {"prev": [23], "next": []},
            ],
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            [
                {"prev": [], "next": [2]},  # rank = 0. encoder1 stage 0
                {"prev": [], "next": [3]},
                {"prev": [0], "next": [10, 11, 14, 15, 18, 19, 22, 23]},
                {"prev": [1], "next": [12, 13, 16, 17, 20, 21, 24, 25]},
                {"prev": [], "next": [6]},  # rank = 4. encoder2 stage 0
                {"prev": [], "next": [7]},
                {"prev": [4], "next": [8]},
                {"prev": [5], "next": [9]},
                {"prev": [6], "next": [10, 11, 14, 15, 18, 19, 22, 23]},
                {"prev": [7], "next": [12, 13, 16, 17, 20, 21, 24, 25]},
                {"prev": [2, 8], "next": [26]},  # rank = 10, LLM stage 0
                {"prev": [2, 8], "next": [27]},
                {"prev": [3, 9], "next": [28]},
                {"prev": [3, 9], "next": [29]},
                {"prev": [2, 8], "next": [30]},  # rank = 14
                {"prev": [2, 8], "next": [31]},
                {"prev": [3, 9], "next": [32]},
                {"prev": [3, 9], "next": [33]},
                {"prev": [2, 8], "next": [34]},  # rank = 18
                {"prev": [2, 8], "next": [35]},
                {"prev": [3, 9], "next": [36]},
                {"prev": [3, 9], "next": [37]},
                {"prev": [2, 8], "next": [38]},  # rank = 22
                {"prev": [2, 8], "next": [39]},
                {"prev": [3, 9], "next": [40]},
                {"prev": [3, 9], "next": [41]},
                {"prev": [10], "next": []},  # rank = 26. LLM last stage
                {"prev": [11], "next": []},
                {"prev": [12], "next": []},
                {"prev": [13], "next": []},
                {"prev": [14], "next": []},  # rank = 30
                {"prev": [15], "next": []},
                {"prev": [16], "next": []},
                {"prev": [17], "next": []},
                {"prev": [18], "next": []},  # rank = 34
                {"prev": [19], "next": []},
                {"prev": [20], "next": []},
                {"prev": [21], "next": []},
                {"prev": [22], "next": []},  # rank = 38
                {"prev": [23], "next": []},
                {"prev": [24], "next": []},
                {"prev": [25], "next": []},
            ],
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            [
                {"prev": [], "next": [4]},  # rank = 0. encoder2 stage 0
                {"prev": [], "next": [5]},
                {"prev": [], "next": [6]},
                {"prev": [], "next": [7]},
                {"prev": [0], "next": [8]},  # rank = 4. encoder2 stage 1
                {"prev": [1], "next": [9]},
                {"prev": [2], "next": [10]},
                {"prev": [3], "next": [11]},
                {"prev": [4], "next": [12, 14, 16, 18]},  # rank = 8. encoder2 last → llm
                {"prev": [5], "next": [12, 14, 16, 18]},
                {"prev": [6], "next": [13, 15, 17, 19]},
                {"prev": [7], "next": [13, 15, 17, 19]},
                {"prev": [8, 9], "next": [20]},  # rank = 12, LLM stage 0
                {"prev": [10, 11], "next": [21]},
                {"prev": [8, 9], "next": [22]},
                {"prev": [10, 11], "next": [23]},
                {"prev": [8, 9], "next": [24]},  # rank = 16
                {"prev": [10, 11], "next": [25]},
                {"prev": [8, 9], "next": [26]},
                {"prev": [10, 11], "next": [27]},
                {"prev": [12], "next": [28]},  # rank = 20
                {"prev": [13], "next": [29]},
                {"prev": [14], "next": [30]},
                {"prev": [15], "next": [31]},
                {"prev": [16], "next": [32]},  # rank = 24
                {"prev": [17], "next": [33]},
                {"prev": [18], "next": [34]},
                {"prev": [19], "next": [35]},
                {"prev": [20], "next": [36]},  # rank = 28
                {"prev": [21], "next": [37]},
                {"prev": [22], "next": [38]},
                {"prev": [23], "next": [39]},
                {"prev": [24], "next": [40]},  # rank = 32
                {"prev": [25], "next": [41]},
                {"prev": [26], "next": [42]},
                {"prev": [27], "next": [43]},
                {"prev": [28], "next": []},  # rank = 36. LLM last stage
                {"prev": [29], "next": []},
                {"prev": [30], "next": []},
                {"prev": [31], "next": []},
                {"prev": [32], "next": []},  # rank = 40
                {"prev": [33], "next": []},
                {"prev": [34], "next": []},
                {"prev": [35], "next": []},
            ],
        ),
        # ------------------------------------------------------------------
        # Case A: encoder TP=4 SP=1, LLM TP=2 SP=1 (only TP different,
        #          encoder TP > LLM TP), DP=1, world_size=12
        # encoder mesh [2,1,1,4]: stage0=[0,1,2,3], stage1=[4,5,6,7]
        # LLM    mesh [2,1,1,2]: stage0=[8,9],      stage1=[10,11]
        # Border: encoder→LLM fan-in 2:1 on TP
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 4},
            (llm_template_2stages, 2, 1),
            [
                {"prev": [], "next": [4]},   # rank 0. encoder stage 0
                {"prev": [], "next": [5]},
                {"prev": [], "next": [6]},
                {"prev": [], "next": [7]},
                {"prev": [0], "next": [8]},  # rank 4. encoder stage 1
                {"prev": [1], "next": [8]},
                {"prev": [2], "next": [9]},
                {"prev": [3], "next": [9]},
                {"prev": [4, 5], "next": [10]},  # rank 8. LLM stage 0
                {"prev": [6, 7], "next": [11]},
                {"prev": [8], "next": []},       # rank 10. LLM last stage
                {"prev": [9], "next": []},
            ],
        ),
        # ------------------------------------------------------------------
        # Case B: encoder TP=2 SP=1, LLM TP=2 SP=2 (only SP different,
        #          encoder SP < LLM SP), DP=1, world_size=12
        # encoder mesh [2,1,1,2]: stage0=[0,1], stage1=[2,3]
        # LLM    mesh [2,1,2,2]: stage0 SP0=[4,5] SP1=[6,7];
        #                        stage1 SP0=[8,9] SP1=[10,11]
        # Border: encoder last → LLM first fan-out 1:2 on SP
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            [
                {"prev": [], "next": [2]},         # rank 0. encoder stage 0
                {"prev": [], "next": [3]},
                {"prev": [0], "next": [4, 6]},     # rank 2. encoder last → llm
                {"prev": [1], "next": [5, 7]},
                {"prev": [2], "next": [8]},         # rank 4. LLM stage 0 SP=0
                {"prev": [3], "next": [9]},
                {"prev": [2], "next": [10]},        # rank 6. LLM stage 0 SP=1
                {"prev": [3], "next": [11]},
                {"prev": [4], "next": []},           # rank 8. LLM last stage
                {"prev": [5], "next": []},
                {"prev": [6], "next": []},
                {"prev": [7], "next": []},
            ],
        ),
        # ------------------------------------------------------------------
        # Case C: encoder TP=2 SP=2, LLM TP=2 SP=1 (only SP different,
        #          encoder SP > LLM SP), DP=1, world_size=12
        # encoder mesh [2,1,2,2]: stage0 SP0=[0,1] SP1=[2,3];
        #                         stage1 SP0=[4,5] SP1=[6,7]
        # LLM    mesh [2,1,1,2]: stage0=[8,9], stage1=[10,11]
        # Border: encoder last → LLM first fan-in 2:1 on SP
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 1),
            [
                {"prev": [], "next": [4]},   # rank 0. encoder stage 0 SP=0 TP=0
                {"prev": [], "next": [5]},
                {"prev": [], "next": [6]},   # rank 2. encoder stage 0 SP=1 TP=0
                {"prev": [], "next": [7]},
                {"prev": [0], "next": [8]},  # rank 4. encoder stage 1 SP=0 TP=0
                {"prev": [1], "next": [9]},
                {"prev": [2], "next": [8]},  # rank 6. encoder stage 1 SP=1 TP=0
                {"prev": [3], "next": [9]},
                {"prev": [4, 6], "next": [10]},  # rank 8. LLM stage 0
                {"prev": [5, 7], "next": [11]},
                {"prev": [8], "next": []},       # rank 10. LLM last stage
                {"prev": [9], "next": []},
            ],
        ),
        # ------------------------------------------------------------------
        # Case D: encoder TP=4 SP=2, LLM TP=2 SP=1 (both different,
        #          encoder TP > LLM TP AND encoder SP > LLM SP), DP=1,
        #          world_size=20
        # encoder mesh [2,1,2,4]:
        #   stage0 SP0=[0..3] SP1=[4..7]; stage1 SP0=[8..11] SP1=[12..15]
        # LLM    mesh [2,1,1,2]: stage0=[16,17], stage1=[18,19]
        # Border: encoder last → LLM first fan-in 4:1 on TP×SP
        # ------------------------------------------------------------------
        (
            20,
            {encoder1_template: (4, 2)},
            (llm_template_2stages, 2, 1),
            [
                {"prev": [], "next": [8]},    # rank 0. encoder stage 0 SP=0 TP=0
                {"prev": [], "next": [9]},
                {"prev": [], "next": [10]},
                {"prev": [], "next": [11]},
                {"prev": [], "next": [12]},   # rank 4. encoder stage 0 SP=1 TP=0
                {"prev": [], "next": [13]},
                {"prev": [], "next": [14]},
                {"prev": [], "next": [15]},
                {"prev": [0], "next": [16]},  # rank 8. encoder stage 1 SP=0 TP=0
                {"prev": [1], "next": [16]},
                {"prev": [2], "next": [17]},
                {"prev": [3], "next": [17]},
                {"prev": [4], "next": [16]},  # rank 12. encoder stage 1 SP=1 TP=0
                {"prev": [5], "next": [16]},
                {"prev": [6], "next": [17]},
                {"prev": [7], "next": [17]},
                {"prev": [8, 9, 12, 13], "next": [18]},   # rank 16. LLM stage 0
                {"prev": [10, 11, 14, 15], "next": [19]},
                {"prev": [16], "next": []},  # rank 18. LLM last stage
                {"prev": [17], "next": []},
            ],
        ),
    ],
)
def test_multimodal_pipeline_stage_manager(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

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
    "world_size, encoder_templates, llm_template, expected_first_last_stages, expected_first_last_stages_in_modal",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1, 2, 3): (True, False),
                tuple(range(4, 16)): (False, False),
                tuple(range(16, 24)): (False, True),
            },
            {
                (0, 1, 2, 3): (True, False),
                (4, 5, 6, 7): (False, True),
                tuple(range(8, 16)): (True, False),
                tuple(range(16, 24)): (False, True),
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1, 4, 5): (True, False),
                (2, 3, 6, 7) + tuple(range(8, 14)): (False, False),
                (14, 15, 16, 17): (False, True),
            },
            {
                (0, 1, 4, 5): (True, False),
                (6, 7): (False, False),
                (2, 3, 8, 9): (False, True),
                (10, 11, 12, 13): (True, False),
                (14, 15, 16, 17): (False, True),
            },
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                tuple(range(0, 12)): (True, False),
                tuple(range(12, 72)): (False, False),
                tuple(range(72, 84)): (False, True),
            },
            {
                tuple(range(0, 12)): (True, False),
                tuple(range(12, 24)): (False, False),
                tuple(range(24, 36)): (False, True),
                tuple(range(36, 48)): (True, False),
                tuple(range(48, 72)): (False, False),
                tuple(range(72, 84)): (False, True),
            },
        ),
        (
            40,  # 20 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                tuple(range(0, 4)): (True, False),
                tuple(range(4, 24)): (False, False),
                tuple(range(24, 40)): (False, True),
            },
            {
                tuple(range(0, 4)): (True, False),
                tuple(range(4, 8)): (False, True),
                tuple(range(8, 24)): (True, False),
                tuple(range(24, 40)): (False, True),
            },
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                (0, 1, 4, 5): (True, False),
                (2, 3, 6, 7, 8, 9) + tuple(range(10, 26)): (False, False),
                tuple(range(26, 42)): (False, True),
            },
            {
                (0, 1, 4, 5): (True, False),
                (6, 7): (False, False),
                (2, 3, 8, 9): (False, True),
                tuple(range(10, 26)): (True, False),
                tuple(range(26, 42)): (False, True),
            },
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                tuple(range(0, 4)): (True, False),
                tuple(range(4, 36)): (False, False),
                tuple(range(36, 44)): (False, True),
            },
            {
                tuple(range(0, 4)): (True, False),
                tuple(range(4, 8)): (False, False),
                tuple(range(8, 12)): (False, True),
                tuple(range(12, 20)): (True, False),
                tuple(range(20, 36)): (False, False),
                tuple(range(36, 44)): (False, True),
            },
        ),
        # ------------------------------------------------------------------
        # Case A: encoder TP=4 SP=1 (PP=2), LLM TP=2 SP=1 (PP=2), DP=1
        # encoder ranks 0-7; LLM ranks 8-11
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 4},
            (llm_template_2stages, 2, 1),
            {
                # global: encoder stage 0 = first, LLM stage 1 = last
                (0, 1, 2, 3): (True, False),
                (4, 5, 6, 7, 8, 9): (False, False),
                (10, 11): (False, True),
            },
            {
                # modal-local: encoder stage 0 = first, encoder stage 1 = last;
                #              LLM stage 0 = first (in modal), LLM stage 1 = last
                (0, 1, 2, 3): (True, False),
                (4, 5, 6, 7): (False, True),
                (8, 9): (True, False),
                (10, 11): (False, True),
            },
        ),
        # ------------------------------------------------------------------
        # Case B: encoder TP=2 SP=1 (PP=2), LLM TP=2 SP=2 (PP=2), DP=1
        # encoder ranks 0-3; LLM ranks 4-11
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            {
                (0, 1): (True, False),
                (2, 3, 4, 5, 6, 7): (False, False),
                (8, 9, 10, 11): (False, True),
            },
            {
                (0, 1): (True, False),
                (2, 3): (False, True),
                (4, 5, 6, 7): (True, False),
                (8, 9, 10, 11): (False, True),
            },
        ),
        # ------------------------------------------------------------------
        # Case C: encoder TP=2 SP=2 (PP=2), LLM TP=2 SP=1 (PP=2), DP=1
        # encoder ranks 0-7; LLM ranks 8-11
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 1),
            {
                (0, 1, 2, 3): (True, False),
                (4, 5, 6, 7, 8, 9): (False, False),
                (10, 11): (False, True),
            },
            {
                (0, 1, 2, 3): (True, False),
                (4, 5, 6, 7): (False, True),
                (8, 9): (True, False),
                (10, 11): (False, True),
            },
        ),
        # ------------------------------------------------------------------
        # Case D: encoder TP=4 SP=2 (PP=2), LLM TP=2 SP=1 (PP=2), DP=1
        # encoder ranks 0-15; LLM ranks 16-19
        # ------------------------------------------------------------------
        (
            20,
            {encoder1_template: (4, 2)},
            (llm_template_2stages, 2, 1),
            {
                tuple(range(0, 8)): (True, False),
                tuple(range(8, 18)): (False, False),
                (18, 19): (False, True),
            },
            {
                tuple(range(0, 8)): (True, False),
                tuple(range(8, 16)): (False, True),
                (16, 17): (True, False),
                (18, 19): (False, True),
            },
        ),
    ],
)
# expected_first_last_stage: dict of list of ranks -> tuple of expected (is_first_stage, is_last_stage)
def test_first_last_stage(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_first_last_stages: dict[tuple[int], tuple[bool, bool]],
    expected_first_last_stages_in_modal: dict[tuple[int], tuple[bool, bool]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

        # check modal-local stage
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

        # check global stage
        expected_first_last_stage_in_modal = next(
            value
            for ranks, value in expected_first_last_stages_in_modal.items()
            if rank in ranks
        )
        assert expected_first_last_stage_in_modal == (
            stage_manager.is_first_stage(check_only_in_modal=True),
            stage_manager.is_last_stage(check_only_in_modal=True),
        ), (
            f"rank {rank} expected to have {expected_first_last_stage_in_modal} as first and last stage in modal, "
            f"but got ({stage_manager.is_first_stage(check_only_in_modal=True), stage_manager.is_last_stage(check_only_in_modal=True)})."
        )

        # check automatic behavior, which should be the same with check_only_in_modal=True
        assert expected_first_last_stage_in_modal == (
            stage_manager.is_first_stage(),
            stage_manager.is_last_stage(),
        ), (
            f"rank {rank} expected to have {expected_first_last_stage_in_modal} as first and last stage in modal, "
            f"but got ({stage_manager.is_first_stage(), stage_manager.is_last_stage()})."
        )

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_ranks_in_stage",
    [
        # ------------------------------------------------------------------
        # encoder TP=2 SP=1 (PP=2), LLM TP=4 SP=1 (PP=2), DP=2
        # Stage indices are modal-local (0 = first stage of ANY modal).
        # max valid index = min(PP across all modals) - 1 = 1
        # ------------------------------------------------------------------
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): [
                    # encoder1 PP groups: stage0=[0..3] × stage1=[4..7]
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    # LLM PP groups (DP=2, TP=4): stage0=[8..15] × stage1=[16..23]
                    [8, 16], [9, 17], [10, 18], [11, 19],
                    [12, 20], [13, 21], [14, 22], [15, 23],
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder1 TP=2 (PP=2), encoder2 TP=2 (PP=3), LLM TP=4 (PP=2), DP=1
        # max valid index = 1 (limited by PP=2 modals)
        # ------------------------------------------------------------------
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): [
                    # encoder1: stage0=[0,1] stage1=[2,3]
                    [0, 2], [1, 3],
                    # encoder2: stage0=[4,5] stage1=[6,7]
                    [4, 6], [5, 7],
                    # LLM: stage0=[10..13] stage1=[14..17]
                    [10, 14], [11, 15], [12, 16], [13, 17],
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder2 TP=4 (PP=3), LLM TP=4 (PP=4), DP=3
        # max valid index = 2 (limited by encoder2 PP=3)
        # ------------------------------------------------------------------
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                (0, 1): [
                    # encoder2 (PP=3, DP=3, TP=4): 12 stage0+1 pairs
                    [0, 12], [1, 13], [2, 14], [3, 15],
                    [4, 16], [5, 17], [6, 18], [7, 19],
                    [8, 20], [9, 21], [10, 22], [11, 23],
                    # LLM (PP=4, DP=3, TP=4): 12 stage0+1 pairs
                    [36, 48], [37, 49], [38, 50], [39, 51],
                    [40, 52], [41, 53], [42, 54], [43, 55],
                    [44, 56], [45, 57], [46, 58], [47, 59],
                ],
                (0, 1, 2): [
                    # encoder2: all 3 stages
                    [0, 12, 24], [1, 13, 25], [2, 14, 26], [3, 15, 27],
                    [4, 16, 28], [5, 17, 29], [6, 18, 30], [7, 19, 31],
                    [8, 20, 32], [9, 21, 33], [10, 22, 34], [11, 23, 35],
                    # LLM stages 0+1+2 (stage 3 excluded)
                    [36, 48, 60], [37, 49, 61], [38, 50, 62], [39, 51, 63],
                    [40, 52, 64], [41, 53, 65], [42, 54, 66], [43, 55, 67],
                    [44, 56, 68], [45, 57, 69], [46, 58, 70], [47, 59, 71],
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder TP=2 SP=1 (PP=2), LLM TP=4 SP=2 (PP=2), DP=2
        # max valid index = 1
        # LLM mesh [2,2,2,4]: stage0=[8..23], stage1=[24..39]
        # ------------------------------------------------------------------
        (
            40,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                (0, 1): [
                    # encoder1 (PP=2, DP=2, SP=1, TP=2): 4 pairs
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    # LLM (PP=2, DP=2, SP=2, TP=4): 16 pairs
                    [8, 24], [9, 25], [10, 26], [11, 27],
                    [12, 28], [13, 29], [14, 30], [15, 31],
                    [16, 32], [17, 33], [18, 34], [19, 35],
                    [20, 36], [21, 37], [22, 38], [23, 39],
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder1 TP=2 (PP=2), encoder2 TP=2 (PP=3), LLM TP=4 SP=4 (PP=2), DP=1
        # max valid index = 1
        # LLM mesh [2,1,4,4]: stage0=[10..25], stage1=[26..41]
        # ------------------------------------------------------------------
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                (0, 1): [
                    # encoder1: stage0=[0,1] stage1=[2,3]
                    [0, 2], [1, 3],
                    # encoder2: stage0=[4,5] stage1=[6,7]
                    [4, 6], [5, 7],
                    # LLM (PP=2, SP=4, TP=4): 16 stage0+1 pairs
                    [10, 26], [11, 27], [12, 28], [13, 29],
                    [14, 30], [15, 31], [16, 32], [17, 33],
                    [18, 34], [19, 35], [20, 36], [21, 37],
                    [22, 38], [23, 39], [24, 40], [25, 41],
                ],
            },
        ),
        # ------------------------------------------------------------------
        # encoder2 TP=4 (PP=3), LLM TP=2 SP=4 (PP=4), DP=1
        # max valid index = 2 (encoder2 PP=3 is the limiting modal)
        # encoder2 mesh [3,1,1,4]: stage0=[0..3], stage1=[4..7], stage2=[8..11]
        # LLM     mesh [4,1,4,2]: stage0=[12..19], stage1=[20..27],
        #                          stage2=[28..35], stage3=[36..43]
        # ------------------------------------------------------------------
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                (0, 1): [
                    # encoder2: stage0+1
                    [0, 4], [1, 5], [2, 6], [3, 7],
                    # LLM: stage0+1 (SP=4, TP=2 → 8 pairs per stage pair)
                    [12, 20], [13, 21], [14, 22], [15, 23],
                    [16, 24], [17, 25], [18, 26], [19, 27],
                ],
                (0, 1, 2): [
                    # encoder2: all 3 stages
                    [0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11],
                    # LLM: stages 0+1+2
                    [12, 20, 28], [13, 21, 29], [14, 22, 30], [15, 23, 31],
                    [16, 24, 32], [17, 25, 33], [18, 26, 34], [19, 27, 35],
                ],
            },
        ),
    ],
)
# expected_ranks_in_stage: modal-local stage indices → list of [rank…] per group
def test_process_group_by_stages(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_ranks_in_stage: dict[tuple[int], list[list[int]]],
    mocker: MockerFixture,
):
    recorded_new_group_calls: dict[int, list] = defaultdict(list)
    group_by_stages: dict[int, set[tuple[int]]] = defaultdict(set)

    def record_new_group_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # dist.new_group is called with ranks= as a keyword argument
            ranks = args[0] if args else kwargs.get("ranks", [])
            recorded_new_group_calls[dist.get_rank()].append(tuple(sorted(ranks)))
            return func(*args, **kwargs)

        return wrapper

    mocker.patch.object(
        dist,
        "new_group",
        wraps=record_new_group_call_decorator(dist.new_group),
    )

    for stage_indices, expected_ranks in expected_ranks_in_stage.items():
        for rank in range(world_size):
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
            stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)
            groups = stage_manager.init_process_group_by_stages(stage_indices)

            if not isinstance(groups, list):
                groups = [groups]

            for group in groups:
                if group == dist.GroupMember.NON_GROUP_MEMBER or group is None:
                    continue

                group_by_stages[rank].add(tuple(dist.get_process_group_ranks(group)))

            # check the ranks in the group are as expected
            expected_ranks_with_rank = set(
                tuple(ranks) for ranks in expected_ranks if rank in ranks
            )
            assert group_by_stages[rank] == expected_ranks_with_rank

            dist.destroy_process_group()

        # check new_group call order is all the same across all ranks
        for rank, calls in recorded_new_group_calls.items():
            assert calls == recorded_new_group_calls[0]

        group_by_stages.clear()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_stage_index",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1, 2, 3): (0, 0),
                (4, 5, 6, 7): (1, 1),
                (8, 9, 10, 11, 12, 13, 14, 15): (0, 0),   # LLM stage 0 (modal-local)
                (16, 17, 18, 19, 20, 21, 22, 23): (1, 1),  # LLM stage 1 (modal-local)
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): (0, 0),   # encoder1 stage 0
                (2, 3): (1, 1),   # encoder1 stage 1
                (4, 5): (0, 0),   # encoder2 stage 0 (modal-local)
                (6, 7): (1, 1),   # encoder2 stage 1 (modal-local)
                (8, 9): (2, 2),   # encoder2 stage 2 (modal-local)
                (10, 11, 12, 13): (0, 0),  # LLM stage 0 (modal-local)
                (14, 15, 16, 17): (1, 1),  # LLM stage 1 (modal-local)
            },
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                tuple(range(0, 12)): (0, 0),
                tuple(range(12, 24)): (1, 1),
                tuple(range(24, 36)): (2, 2),
                tuple(range(36, 48)): (0, 0),   # LLM stage 0 (modal-local)
                tuple(range(48, 60)): (1, 1),   # LLM stage 1 (modal-local)
                tuple(range(60, 72)): (2, 2),   # LLM stage 2 (modal-local)
                tuple(range(72, 84)): (3, 3),   # LLM stage 3 (modal-local)
            },
        ),
        # ------------------------------------------------------------------
        # Case A: encoder TP=4 SP=1, LLM TP=2 SP=1, world_size=12
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 4},
            (llm_template_2stages, 2, 1),
            {
                (0, 1, 2, 3): (0, 0),   # encoder stage 0
                (4, 5, 6, 7): (1, 1),   # encoder stage 1
                (8, 9): (0, 0),          # LLM stage 0 (modal-local)
                (10, 11): (1, 1),        # LLM stage 1 (modal-local)
            },
        ),
        # ------------------------------------------------------------------
        # Case B: encoder TP=2 SP=1, LLM TP=2 SP=2, world_size=12
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            {
                (0, 1): (0, 0),    # encoder stage 0
                (2, 3): (1, 1),    # encoder stage 1
                (4, 5, 6, 7): (0, 0),    # LLM stage 0 (modal-local)
                (8, 9, 10, 11): (1, 1),  # LLM stage 1 (modal-local)
            },
        ),
        # ------------------------------------------------------------------
        # Case C: encoder TP=2 SP=2, LLM TP=2 SP=1, world_size=12
        # ------------------------------------------------------------------
        (
            12,
            {encoder1_template: (2, 2)},
            (llm_template_2stages, 2, 1),
            {
                (0, 1, 2, 3): (0, 0),   # encoder stage 0
                (4, 5, 6, 7): (1, 1),   # encoder stage 1
                (8, 9): (0, 0),          # LLM stage 0 (modal-local)
                (10, 11): (1, 1),        # LLM stage 1 (modal-local)
            },
        ),
        # ------------------------------------------------------------------
        # Case D: encoder TP=4 SP=2, LLM TP=2 SP=1, world_size=20
        # ------------------------------------------------------------------
        (
            20,
            {encoder1_template: (4, 2)},
            (llm_template_2stages, 2, 1),
            {
                (0, 1, 2, 3, 4, 5, 6, 7): (0, 0),   # encoder stage 0
                (8, 9, 10, 11, 12, 13, 14, 15): (1, 1),  # encoder stage 1
                (16, 17): (0, 0),  # LLM stage 0 (modal-local)
                (18, 19): (1, 1),  # LLM stage 1 (modal-local)
            },
        ),
    ],
)
def test_stage(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_stage_index: dict[tuple[int, ...], tuple[int, int]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)
        expected_stage_index_for_rank = next(
            value for ranks, value in expected_stage_index.items() if rank in ranks
        )
        assert (
            stage_manager.stage,
            stage_manager.stage_in_modal,
        ) == expected_stage_index_for_rank, (
            f"rank {rank} expected: {expected_stage_index_for_rank}, "
            f"got: {stage_manager.stage, stage_manager.stage_in_modal}."
        )

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_layer_distributions, expected_stage_index_per_modal",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            # Per-modal layers-per-stage (modal-local)
            {"encoder1": [2, 2], "llm": [3, 2]},
            {
                (0, 1, 2, 3, 4, 5, 6, 7): {  # encoder1
                    (0,): (0, 2),
                    (1,): (2, 4),
                },
                tuple(range(8, 24)): {  # llm
                    (0,): (0, 3),
                    (1,): (3, 5),
                },
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {"encoder1": [2, 2], "encoder2": [2, 2, 2], "llm": [3, 2]},
            {
                (0, 1, 2, 3): {  # encoder1
                    (0,): (0, 2),
                    (1,): (2, 4),
                },
                (4, 5, 6, 7, 8, 9): {  # encoder2
                    (0,): (0, 2),
                    (1,): (2, 4),
                    (2,): (4, 6),
                },
                tuple(range(10, 18)): {  # llm
                    (0,): (0, 3),
                    (1,): (3, 5),
                },
            },
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {"encoder2": [2, 2, 2], "llm": [3, 1, 4, 2]},
            {
                tuple(range(0, 12)): {  # encoder2
                    (0,): (0, 2),
                    (1,): (2, 4),
                    (2,): (4, 6),
                },
                tuple(range(12, 44)): {  # llm
                    (0,): (0, 3),
                    (1,): (3, 4),
                    (2,): (4, 8),
                    (3,): (8, 10),
                },
            },
        ),
    ],
)
def test_layer_distribution(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_layer_distributions: dict[str, list[int]],
    # dict of rank-tuple → dict of modal-local stage index tuple → (start, end)
    expected_stage_index_per_modal: dict[
        tuple[int, ...], dict[tuple[int, ...], tuple[int, int]]
    ],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

        layers = stage_manager.distribute_layers()
        modal_name = mesh.my_modal.model_name
        expected_layers = expected_layer_distributions[modal_name]
        assert (
            layers == expected_layers
        ), f"rank {rank} ({modal_name}) layer distribution expected: {expected_layers}, got: {layers}."

        expected_stage_indices_for_rank = next(
            value
            for ranks, value in expected_stage_index_per_modal.items()
            if rank in ranks
        )

        for stage_index in range(len(layers)):
            expected_layer_indices = next(
                value
                for stage_indices, value in expected_stage_indices_for_rank.items()
                if stage_index in stage_indices
            )
            layer_indices = stage_manager.get_stage_index(layers, stage=stage_index)
            assert (
                layer_indices == expected_layer_indices
            ), f"rank {rank} expected stage index: {expected_layer_indices}, got: {layer_indices}."

        dist.destroy_process_group()
