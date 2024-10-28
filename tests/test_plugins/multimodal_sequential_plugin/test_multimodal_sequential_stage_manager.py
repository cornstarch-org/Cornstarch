import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_sequential_plugin.multimodal_sequential_stage_manager import (
    MultimodalSequentialPipelineStageManager,
)
from cornstarch.plugin.multimodal_sequential_plugin.process_group_mesh import (
    MultimodalSequentialProcessGroupMesh,
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
    "world_size, encoder_templates, llm_template, expected_prev_next_ranks",
    [
        (
            4,
            {encoder1_template: 1},
            (llm_template_2stages, 1, 1),
            [
                {"prev": [3], "next": [1]},
                {"prev": [0], "next": [2]},
                {"prev": [1], "next": [3]},
                {"prev": [2], "next": [0]},
            ],
        ),
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
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            [
                {"prev": [8, 9], "next": [2]},
                {"prev": [10, 11], "next": [3]},
                {"prev": [0], "next": [4, 5]},
                {"prev": [1], "next": [6, 7]},
                {"prev": [2], "next": [8]},  # rank = 4. llm
                {"prev": [2], "next": [9]},
                {"prev": [3], "next": [10]},
                {"prev": [3], "next": [11]},
                {"prev": [4], "next": [0]},  # rank = 8
                {"prev": [5], "next": [0]},
                {"prev": [6], "next": [1]},
                {"prev": [7], "next": [1]},
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
                {"prev": [0], "next": [8, 9, 12, 13]},  # rank 4
                {"prev": [1], "next": [10, 11, 14, 15]},
                {"prev": [2], "next": [16, 17, 20, 21]},
                {"prev": [3], "next": [18, 19, 22, 23]},
                {"prev": [4], "next": [24]},  # rank 8. llm
                {"prev": [4], "next": [25]},
                {"prev": [5], "next": [26]},
                {"prev": [5], "next": [27]},
                {"prev": [4], "next": [28]},  # rank 12
                {"prev": [4], "next": [29]},
                {"prev": [5], "next": [30]},
                {"prev": [5], "next": [31]},
                {"prev": [6], "next": [32]},  # rank 16
                {"prev": [6], "next": [33]},
                {"prev": [7], "next": [34]},
                {"prev": [7], "next": [35]},
                {"prev": [6], "next": [36]},  # rank 20
                {"prev": [6], "next": [37]},
                {"prev": [7], "next": [38]},
                {"prev": [7], "next": [39]},
                {"prev": [8], "next": [0]},  # rank 24
                {"prev": [9], "next": [0]},
                {"prev": [10], "next": [1]},
                {"prev": [11], "next": [1]},
                {"prev": [12], "next": [0]},  # rank 28
                {"prev": [13], "next": [0]},
                {"prev": [14], "next": [1]},
                {"prev": [15], "next": [1]},
                {"prev": [16], "next": [2]},  # rank 32
                {"prev": [17], "next": [2]},
                {"prev": [18], "next": [3]},
                {"prev": [19], "next": [3]},
                {"prev": [20], "next": [2]},  # rank 36
                {"prev": [21], "next": [2]},
                {"prev": [22], "next": [3]},
                {"prev": [23], "next": [3]},
            ],
        ),
    ],
)
def test_multimodal_sequential_pipeline_stage_manager(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)

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
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): (True, False),
                (2, 3): (False, False),
                (4, 5, 6, 7): (False, False),
                (8, 9, 10, 11): (False, True),
            },
            {
                (0, 1): (True, False),
                (2, 3): (False, True),
                (4, 5, 6, 7): (True, False),
                (8, 9, 10, 11): (False, True),
            },
        ),
        (
            72,  # 36 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_4stages, 4, 2),
            {
                (0, 1, 2, 3): (True, False),  # encoder stage 0
                (4, 5, 6, 7): (False, False),  # encoder stage 1
                tuple(range(8, 24)): (False, False),  # llm stage 0
                tuple(range(24, 56)): (False, False),  # llm stages 1~2
                tuple(range(56, 72)): (False, True),  # llm stage 3
            },
            {
                (0, 1, 2, 3): (True, False),  # encoder stage 0
                (4, 5, 6, 7): (False, True),  # encoder stage 1
                tuple(range(8, 24)): (True, False),  # llm stage 0
                tuple(range(24, 56)): (False, False),  # llm stages 1~2
                tuple(range(56, 72)): (False, True),  # llm stage 3
            },
        ),
        (
            80,  # 40 ranks * 2 dp
            {encoder2_template: 4},
            (llm_template_2stages, 4, 1),
            {
                tuple(range(0, 16)): (True, False),  # encoder stage 0
                tuple(range(16, 32)): (False, False),  # encoder stage 1
                tuple(range(32, 48)): (False, False),  # encoder stage 2
                tuple(range(48, 64)): (False, False),  # llm stage 0
                tuple(range(64, 80)): (False, True),  # llm stage 1
            },
            {
                tuple(range(0, 16)): (True, False),  # encoder stage 0
                tuple(range(16, 32)): (False, False),  # encoder stage 1
                tuple(range(32, 48)): (False, True),  # encoder stage 2
                tuple(range(48, 64)): (True, False),  # llm stage 0
                tuple(range(64, 80)): (False, True),  # llm stage 1
            },
        ),
    ],
)
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
        mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)

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
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): [[0, 2], [1, 3]],
                (0, 2): [[0, 4], [0, 5], [1, 6], [1, 7]],
                (1, 2): [[2, 4], [2, 5], [3, 6], [3, 7]],
                (0, 1, 3): [[0, 2, 8], [0, 2, 9], [1, 3, 10], [1, 3, 11]],
            },
        ),
        (
            72,  # 36 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_4stages, 4, 2),
            {
                (0, 1): [[0, 4], [1, 5], [2, 6], [3, 7]],
                (0, 3): [
                    [0, 24],
                    [0, 25],
                    [1, 26],
                    [1, 27],
                    [0, 28],
                    [0, 29],
                    [1, 30],
                    [1, 31],
                    [2, 32],
                    [2, 33],
                    [3, 34],
                    [3, 35],
                    [2, 36],
                    [2, 37],
                    [3, 38],
                    [3, 39],
                ],
                (1, 2): [
                    [4, 8],
                    [4, 9],
                    [5, 10],
                    [5, 11],
                    [4, 12],
                    [4, 13],
                    [5, 14],
                    [5, 15],
                    [6, 16],
                    [6, 17],
                    [7, 18],
                    [7, 19],
                    [6, 20],
                    [6, 21],
                    [7, 22],
                    [7, 23],
                ],
                (0, 2, 3): [
                    [0, 8, 24],
                    [0, 9, 25],
                    [1, 10, 26],
                    [1, 11, 27],
                    [0, 12, 28],
                    [0, 13, 29],
                    [1, 14, 30],
                    [1, 15, 31],
                    [2, 16, 32],
                    [2, 17, 33],
                    [3, 18, 34],
                    [3, 19, 35],
                    [2, 20, 36],
                    [2, 21, 37],
                    [3, 22, 38],
                    [3, 23, 39],
                ],
            },
        ),
    ],
)
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
            mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
            stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)
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
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            {
                (0, 1): (0, 0),
                (2, 3): (1, 1),
                (4, 5, 6, 7): (2, 0),
                (8, 9, 10, 11): (3, 1),
            },
        ),
        (
            72,  # 36 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_4stages, 4, 2),
            {
                (0, 1, 2, 3): (0, 0),
                (4, 5, 6, 7): (1, 1),
                tuple(range(8, 24)): (2, 0),
                tuple(range(24, 40)): (3, 1),
                tuple(range(40, 56)): (4, 2),
                tuple(range(56, 72)): (5, 3),
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
        mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)
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
            {
                encoder1_template: [2, 2],
                llm_template_2stages: [0, 0, 3, 2],
            },
            {
                tuple(range(0, 8)): {  # encoder1
                    encoder1_template: [(0, 2), (2, 4)],
                    llm_template_2stages: [(0, 0), (0, 0)],
                },
                tuple(range(8, 24)): {  # llm
                    encoder1_template: [(0, 0), (0, 0)],
                    llm_template_2stages: [(0, 3), (3, 5)],
                },
            },
        ),
        (
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            {
                encoder1_template: [2, 2],
                encoder3_template: [3, 2],
                llm_template_2stages: [0, 0, 3, 2],
            },
            {
                (0, 1, 2, 3): {  # ranks in encoder1/3 (colocated)
                    encoder1_template: [(0, 2), (2, 4)],
                    encoder3_template: [(0, 3), (3, 5)],
                    llm_template_2stages: [(0, 0), (0, 0)],
                },
                tuple(range(4, 12)): {  # ranks in llm
                    encoder1_template: [(0, 0), (0, 0)],
                    encoder3_template: [(0, 0), (0, 0)],
                    llm_template_2stages: [(0, 3), (3, 5)],
                },
            },
        ),
        (
            44,
            {encoder2_template: 4},
            (llm_template_4stages, 2, 4),
            {
                encoder2_template: [2, 2, 2],
                llm_template_4stages: [0, 0, 0, 3, 1, 4, 2],
            },
            {
                tuple(range(0, 12)): {  # ranks in encoder2
                    encoder2_template: [(0, 2), (2, 4), (4, 6)],
                    llm_template_4stages: [(0, 0), (0, 0), (0, 0), (0, 0)],
                },
                tuple(range(12, 44)): {  # ranks in llm
                    encoder2_template: [(0, 0), (0, 0), (0, 0)],
                    llm_template_4stages: [(0, 3), (3, 4), (4, 8), (8, 10)],
                },
            },
        ),
    ],
)
def test_layer_distribution(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_layer_distribution: dict[PipelineTemplate, list[str]],
    # dict of list of ranks -> dict of stage indices ->
    # (dict of pipeline template -> tuple of start and end layer index)
    expected_stage_index_per_modal: dict[
        tuple[int, ...], dict[PipelineTemplate, list[tuple[int, int]]]
    ],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)

        # Encoders have its own stage manager and layer distribution.
        for encoder in encoder_templates.keys():
            encoder_layers = stage_manager.encoder_stage_managers[
                encoder
            ].distribute_layers()
            assert (
                encoder_layers == expected_layer_distribution[encoder]
            ), f"layer distribution expected: {expected_layer_distribution[encoder]}, got: {encoder_layers}."

        llm_layers = stage_manager.distribute_layers()
        assert (
            llm_layers == expected_layer_distribution[llm_template[0]]
        ), f"layer distribution expected: {expected_layer_distribution[llm_template[0]]}, got: {llm_layers}."

        # Layer indices per stage check
        stage_index = stage_manager.get_stage_index(llm_layers)
        expected_stage_indices_for_rank = next(
            value
            for ranks, value in expected_stage_index_per_modal.items()
            if rank in ranks
        )

        for template, expected_stage_indices in expected_stage_indices_for_rank.items():
            if template in encoder_templates:
                for index, expected_stage_index in enumerate(expected_stage_indices):
                    # if rank is not for encoders, skip
                    if not stage_manager._check_my_rank_in_the_stage(index):
                        continue

                    encoder_stage_manager = stage_manager.encoder_stage_managers[
                        template
                    ]
                    encoder_layers = encoder_stage_manager.distribute_layers()
                    stage_index = encoder_stage_manager.get_stage_index(
                        encoder_layers, stage=index
                    )
                    assert (
                        stage_index == expected_stage_index
                    ), f"rank {rank} expected stage index: {expected_stage_index}, got: {stage_index}."
            else:
                assert template == llm_template[0]
                llm_stage_stage_index = next(iter(encoder_templates.keys())).num_stages
                for index, expected_stage_index in enumerate(expected_stage_indices):
                    stage_index = stage_manager.get_stage_index(
                        llm_layers, stage=llm_stage_stage_index + index
                    )
                    assert (
                        stage_index == expected_stage_index
                    ), f"rank {rank} expected stage index: {expected_stage_index}, got: {stage_index}."

        dist.destroy_process_group()
