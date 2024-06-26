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

from .test_modal_process_group_mesh import (
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
    "world_size, modal_templates, execution_order, expected_prev_next_ranks",
    [
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
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
            {
                encoder1_template: 2,
                encoder2_template: 2,
                llm_template_2stages: 4,
            },
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
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
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
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
    ],
)
def test_multimodal_pipeline_stage_manager(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

        assert stage_manager.prev_ranks == expected_prev_next_ranks[rank]["prev"]
        assert stage_manager.next_ranks == expected_prev_next_ranks[rank]["next"]

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, modal_templates, execution_order, expected_first_last_stages",
    [
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
            {
                (0, 1, 2, 3): (True, False),
                tuple(range(4, 16)): (False, False),
                tuple(range(16, 24)): (False, True),
            },
        ),
        (
            18,
            {
                encoder1_template: 2,
                encoder2_template: 2,
                llm_template_2stages: 4,
            },
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
            {
                (0, 1, 4, 5): (True, False),
                (2, 3) + tuple(range(6, 14)): (False, False),
                (14, 15, 16, 17): (False, True),
            },
        ),
        (
            84,
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
            {
                tuple(range(0, 12)): (True, False),
                tuple(range(12, 72)): (False, False),
                tuple(range(72, 84)): (False, True),
            },
        ),
    ],
)
# expected_first_last_stage: dict of list of ranks -> tuple of expected (is_first_stage, is_last_stage)
def test_first_last_stage(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_first_last_stages: dict[tuple[int], tuple[bool, bool]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

        expected_first_last_stage = next(
            value
            for ranks, value in expected_first_last_stages.items()
            if rank in ranks
        )
        assert expected_first_last_stage == (
            stage_manager.is_first_stage(),
            stage_manager.is_last_stage(),
        ), (
            f"rank {rank} expected to have {expected_first_last_stage} as first and last stage, "
            f"but got ({stage_manager.is_first_stage(), stage_manager.is_last_stage()})."
        )

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, modal_templates, execution_order, expected_ranks_in_stage",
    [
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
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
            {
                encoder1_template: 2,
                encoder2_template: 2,
                llm_template_2stages: 4,
            },
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
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
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
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
    ],
)
# expected_ranks_in_stage: list of stage indices -> list of list of ranks
def test_process_group_by_stages(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
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
            mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
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
