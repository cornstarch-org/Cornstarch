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

from .common import (
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
                {"prev": [16, 17], "next": [4]},
                {"prev": [18, 19], "next": [5]},
                {"prev": [20, 21], "next": [6]},
                {"prev": [22, 23], "next": [7]},
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
                {"prev": [8], "next": [0]},  # rank = 16
                {"prev": [9], "next": [0]},
                {"prev": [10], "next": [1]},
                {"prev": [11], "next": [1]},
                {"prev": [12], "next": [2]},  # rank = 20
                {"prev": [13], "next": [2]},
                {"prev": [14], "next": [3]},
                {"prev": [15], "next": [3]},
            ],
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            [
                {"prev": [14, 15], "next": [2]},  # rank = 0. encoder1
                {"prev": [16, 17], "next": [3]},
                {"prev": [0], "next": [10, 11]},  # rank = 2. connected to llm
                {"prev": [1], "next": [12, 13]},
                {"prev": [14, 15], "next": [6]},  # rank = 4. encoder2
                {"prev": [16, 17], "next": [7]},
                {"prev": [4], "next": [8]},
                {"prev": [5], "next": [9]},
                {"prev": [6], "next": [10, 11]},  # rank = 8. connected to llm
                {"prev": [7], "next": [12, 13]},
                {"prev": [2, 8], "next": [14]},
                {"prev": [2, 8], "next": [15]},
                {"prev": [3, 9], "next": [16]},  # rank = 12
                {"prev": [3, 9], "next": [17]},
                {"prev": [10], "next": [0, 4]},
                {"prev": [11], "next": [0, 4]},
                {"prev": [12], "next": [1, 5]},  # rank = 16
                {"prev": [13], "next": [1, 5]},
            ],
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            [
                {"prev": [72], "next": [12]},
                {"prev": [73], "next": [13]},
                {"prev": [74], "next": [14]},
                {"prev": [75], "next": [15]},
                {"prev": [76], "next": [16]},
                {"prev": [77], "next": [17]},
                {"prev": [78], "next": [18]},
                {"prev": [79], "next": [19]},
                {"prev": [80], "next": [20]},
                {"prev": [81], "next": [21]},
                {"prev": [82], "next": [22]},
                {"prev": [83], "next": [23]},
            ]
            + [{"prev": [i - 12], "next": [i + 12]} for i in range(12, 72)]
            + [
                {"prev": [60], "next": [0]},
                {"prev": [61], "next": [1]},
                {"prev": [62], "next": [2]},
                {"prev": [63], "next": [3]},
                {"prev": [64], "next": [4]},
                {"prev": [65], "next": [5]},
                {"prev": [66], "next": [6]},
                {"prev": [67], "next": [7]},
                {"prev": [68], "next": [8]},
                {"prev": [69], "next": [9]},
                {"prev": [70], "next": [10]},
                {"prev": [71], "next": [11]},
            ],
        ),
        (
            40,  # 20 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            [
                {"prev": [24, 25, 28, 29], "next": [4]},
                {"prev": [26, 27, 30, 31], "next": [5]},
                {"prev": [32, 33, 36, 37], "next": [6]},
                {"prev": [34, 35, 38, 39], "next": [7]},
                {"prev": [0], "next": [8, 9, 12, 13]},  # rank = 4
                {"prev": [1], "next": [10, 11, 14, 15]},
                {"prev": [2], "next": [16, 17, 20, 21]},
                {"prev": [3], "next": [18, 19, 22, 23]},
                {"prev": [4], "next": [24]},  # rank = 8
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
                {"prev": [8], "next": [0]},  # rank = 24
                {"prev": [9], "next": [0]},
                {"prev": [10], "next": [1]},
                {"prev": [11], "next": [1]},
                {"prev": [12], "next": [0]},  # rank = 28
                {"prev": [13], "next": [0]},
                {"prev": [14], "next": [1]},
                {"prev": [15], "next": [1]},
                {"prev": [16], "next": [2]},  # rank = 32
                {"prev": [17], "next": [2]},
                {"prev": [18], "next": [3]},
                {"prev": [19], "next": [3]},
                {"prev": [20], "next": [2]},  # rank = 36
                {"prev": [21], "next": [2]},
                {"prev": [22], "next": [3]},
                {"prev": [23], "next": [3]},
            ],
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            [
                {"prev": [26, 27, 30, 31, 34, 35, 38, 39], "next": [2]},
                {"prev": [28, 29, 32, 33, 36, 37, 40, 41], "next": [3]},
                {"prev": [0], "next": [10, 11, 14, 15, 18, 19, 22, 23]},
                {"prev": [1], "next": [12, 13, 16, 17, 20, 21, 24, 25]},
                {"prev": [26, 27, 30, 31, 34, 35, 38, 39], "next": [6]},  # rank = 4
                {"prev": [28, 29, 32, 33, 36, 37, 40, 41], "next": [7]},
                {"prev": [4], "next": [8]},
                {"prev": [5], "next": [9]},
                {"prev": [6], "next": [10, 11, 14, 15, 18, 19, 22, 23]},
                {"prev": [7], "next": [12, 13, 16, 17, 20, 21, 24, 25]},
                {"prev": [2, 8], "next": [26]},  # rank = 10, LLM
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
                {"prev": [10], "next": [0, 4]},  # rank = 26
                {"prev": [11], "next": [0, 4]},
                {"prev": [12], "next": [1, 5]},
                {"prev": [13], "next": [1, 5]},
                {"prev": [14], "next": [0, 4]},  # rank = 30
                {"prev": [15], "next": [0, 4]},
                {"prev": [16], "next": [1, 5]},
                {"prev": [17], "next": [1, 5]},
                {"prev": [18], "next": [0, 4]},  # rank = 34
                {"prev": [19], "next": [0, 4]},
                {"prev": [20], "next": [1, 5]},
                {"prev": [21], "next": [1, 5]},
                {"prev": [22], "next": [0, 4]},  # rank = 38
                {"prev": [23], "next": [0, 4]},
                {"prev": [24], "next": [1, 5]},
                {"prev": [25], "next": [1, 5]},
            ],
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            [
                {"prev": [36, 38, 40, 42], "next": [4]},
                {"prev": [36, 38, 40, 42], "next": [5]},
                {"prev": [37, 39, 41, 43], "next": [6]},
                {"prev": [37, 39, 41, 43], "next": [7]},
                {"prev": [0], "next": [8]},  # rank = 4
                {"prev": [1], "next": [9]},
                {"prev": [2], "next": [10]},
                {"prev": [3], "next": [11]},
                {"prev": [4], "next": [12, 14, 16, 18]},  # rank = 8
                {"prev": [5], "next": [12, 14, 16, 18]},
                {"prev": [6], "next": [13, 15, 17, 19]},
                {"prev": [7], "next": [13, 15, 17, 19]},
                {"prev": [8, 9], "next": [20]},  # rank = 12, LLM
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
                {"prev": [28], "next": [0, 1]},  # rank = 36
                {"prev": [29], "next": [2, 3]},
                {"prev": [30], "next": [0, 1]},
                {"prev": [31], "next": [2, 3]},
                {"prev": [32], "next": [0, 1]},  # rank = 40
                {"prev": [33], "next": [2, 3]},
                {"prev": [34], "next": [0, 1]},
                {"prev": [35], "next": [2, 3]},
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
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): [[0, 4], [1, 5], [2, 6], [3, 7]],
                (0, 2): [
                    [0, 8],
                    [0, 9],
                    [1, 10],
                    [1, 11],
                    [2, 12],
                    [2, 13],
                    [3, 14],
                    [3, 15],
                ],
                (0, 3): [
                    [0, 16],
                    [0, 17],
                    [1, 18],
                    [1, 19],
                    [2, 20],
                    [2, 21],
                    [3, 22],
                    [3, 23],
                ],
                (1, 2): [
                    [4, 8],
                    [4, 9],
                    [5, 10],
                    [5, 11],
                    [6, 12],
                    [6, 13],
                    [7, 14],
                    [7, 15],
                ],
                (0, 1, 3): [
                    [0, 4, 16],
                    [0, 4, 17],
                    [1, 5, 18],
                    [1, 5, 19],
                    [2, 6, 20],
                    [2, 6, 21],
                    [3, 7, 22],
                    [3, 7, 23],
                ],
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): [[0, 2], [1, 3]],
                (0, 2): [[0, 4], [1, 5]],
                (0, 3): [[0, 6], [1, 7]],
                (1, 2): [[2, 4], [3, 5]],
                (0, 1, 2): [[0, 2, 4], [1, 3, 5]],
                (0, 5): [[0, 10], [0, 11], [1, 12], [1, 13]],
                (4, 5): [[8, 10], [8, 11], [9, 12], [9, 13]],
                (2, 5, 6): [[4, 10, 14], [4, 11, 15], [5, 12, 16], [5, 13, 17]],
                (3, 5, 6): [[6, 10, 14], [6, 11, 15], [7, 12, 16], [7, 13, 17]],
            },
        ),
        (
            84,
            {encoder2_template: 4},
            (llm_template_4stages, 4, 1),
            {
                (0, 1): [
                    [0, 12],
                    [1, 13],
                    [2, 14],
                    [3, 15],
                    [4, 16],
                    [5, 17],
                    [6, 18],
                    [7, 19],
                    [8, 20],
                    [9, 21],
                    [10, 22],
                    [11, 23],
                ],
                (0, 2): [
                    [0, 24],
                    [1, 25],
                    [2, 26],
                    [3, 27],
                    [4, 28],
                    [5, 29],
                    [6, 30],
                    [7, 31],
                    [8, 32],
                    [9, 33],
                    [10, 34],
                    [11, 35],
                ],
                (1, 3): [
                    [12, 36],
                    [13, 37],
                    [14, 38],
                    [15, 39],
                    [16, 40],
                    [17, 41],
                    [18, 42],
                    [19, 43],
                    [20, 44],
                    [21, 45],
                    [22, 46],
                    [23, 47],
                ],
                (4, 5, 6): [
                    [48, 60, 72],
                    [49, 61, 73],
                    [50, 62, 74],
                    [51, 63, 75],
                    [52, 64, 76],
                    [53, 65, 77],
                    [54, 66, 78],
                    [55, 67, 79],
                    [56, 68, 80],
                    [57, 69, 81],
                    [58, 70, 82],
                    [59, 71, 83],
                ],
            },
        ),
        (
            40,  # 20 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            {
                (0, 1): [[0, 4], [1, 5], [2, 6], [3, 7]],
                (1, 2): [
                    [4, 8],
                    [4, 9],
                    [4, 12],
                    [4, 13],
                    [5, 10],
                    [5, 11],
                    [5, 14],
                    [5, 15],
                    [6, 16],
                    [6, 17],
                    [6, 20],
                    [6, 21],
                    [7, 18],
                    [7, 19],
                    [7, 22],
                    [7, 23],
                ],
                (2, 3): [[i, i + 16] for i in range(8, 24)],
            },
        ),
        (
            42,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 4),
            {
                (0, 1): [
                    [0, 2],
                    [1, 3],
                ],
                (1, 4, 5): [
                    [2, 8, 10],
                    [2, 8, 11],
                    [2, 8, 14],
                    [2, 8, 15],
                    [2, 8, 18],
                    [2, 8, 19],
                    [2, 8, 22],
                    [2, 8, 23],
                    [3, 9, 12],
                    [3, 9, 13],
                    [3, 9, 16],
                    [3, 9, 17],
                    [3, 9, 20],
                    [3, 9, 21],
                    [3, 9, 24],
                    [3, 9, 25],
                ],
                (5, 6): [[i, i + 16] for i in range(10, 26)],
            },
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                (0, 1): [[0, 4], [1, 5], [2, 6], [3, 7]],
                (2, 3): [
                    [8, 12],
                    [8, 14],
                    [8, 16],
                    [8, 18],
                    [9, 12],
                    [9, 14],
                    [9, 16],
                    [9, 18],
                    [10, 13],
                    [10, 15],
                    [10, 17],
                    [10, 19],
                    [11, 13],
                    [11, 15],
                    [11, 17],
                    [11, 19],
                ],
            },
        ),
    ],
)
# expected_ranks_in_stage: list of stage indices -> list of list of ranks
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
            recorded_new_group_calls[dist.get_rank()].append(args[0])
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
                (8, 9, 10, 11, 12, 13, 14, 15): (2, 0),
                (16, 17, 18, 19, 20, 21, 22, 23): (3, 1),
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): (0, 0),
                (2, 3): (1, 1),
                (4, 5): (2, 0),
                (6, 7): (3, 1),
                (8, 9): (4, 2),
                (10, 11, 12, 13): (5, 0),
                (14, 15, 16, 17): (6, 1),
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
                tuple(range(36, 48)): (3, 0),
                tuple(range(48, 60)): (4, 1),
                tuple(range(60, 72)): (5, 2),
                tuple(range(72, 84)): (6, 3),
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
    "world_size, encoder_templates, llm_template, expected_layer_distribution, expected_stage_index_per_modal",
    [
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
            [2, 2, 3, 2],
            {
                (0, 1, 2, 3, 4, 5, 6, 7): {  # encoder1
                    (0,): (0, 2),
                    (1,): (2, 4),
                    (2, 3): (0, 0),
                },
                tuple(range(8, 24)): {  # llm
                    (0, 1): (0, 0),
                    (2,): (0, 3),
                    (3,): (3, 5),
                },
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2},
            (llm_template_2stages, 4, 1),
            [2, 2, 2, 2, 2, 3, 2],
            {
                (0, 1, 2, 3): {  # ranks in encoder1
                    (0,): (0, 2),
                    (1,): (2, 4),
                    (2, 3, 4, 5, 6): (0, 0),
                },
                (4, 5, 6, 7, 8, 9): {  # ranks in encoder2
                    (0, 1, 5, 6): (0, 0),
                    (2,): (0, 2),
                    (3,): (2, 4),
                    (4,): (4, 6),
                },
                tuple(range(10, 18)): {  # ranks in llm
                    (0, 1, 2, 3, 4): (0, 0),
                    (5,): (0, 3),
                    (6,): (3, 5),
                },
            },
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            [2, 2, 2, 3, 1, 4, 2],
            {
                tuple(range(0, 12)): {  # ranks in encoder2
                    (0,): (0, 2),
                    (1,): (2, 4),
                    (2,): (4, 6),
                    (3, 4, 5, 6): (0, 0),
                },
                tuple(range(12, 44)): {  # ranks in llm
                    (0, 1, 2): (0, 0),
                    (3,): (0, 3),
                    (4,): (3, 4),
                    (5,): (4, 8),
                    (6,): (8, 10),
                },
            },
        ),
    ],
)
def test_layer_distribution(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_layer_distribution: list[list[str]],
    # dict of list of ranks -> dict of stage indices -> tuple of start and end layer index
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
        assert (
            layers == expected_layer_distribution
        ), f"layer distribution expected: {expected_layer_distribution}, got: {layers}."

        stage_index = stage_manager.get_stage_index(layers)
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
