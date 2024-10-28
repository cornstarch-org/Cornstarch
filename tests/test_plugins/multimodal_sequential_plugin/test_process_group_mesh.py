import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
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
    "world_size, encoder_template, llm_template, expected_mesh",
    [
        (
            # 4 stages, 3 replicas
            12,
            {encoder1_template: 1},
            (llm_template_2stages, 1, 1),
            [
                [[[0]], [[1]], [[2]]],
                [[[3]], [[4]], [[5]]],
                [[[6]], [[7]], [[8]]],
                [[[9]], [[10]], [[11]]],
            ],
        ),
        (
            # 4 stages, 1 replica
            8,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 1),
            [
                [[[0, 1]]],
                [[[2, 3]]],
                [[[4, 5]]],
                [[[6, 7]]],
            ],
        ),
        (
            # 6 stages, 2 replicas
            20,
            {encoder1_template: 1},
            (llm_template_4stages, 2, 1),
            [
                [[[0, 0]], [[1, 1]]],
                [[[2, 2]], [[3, 3]]],
                [[[4, 5]], [[6, 7]]],
                [[[8, 9]], [[10, 11]]],
                [[[12, 13]], [[14, 15]]],
                [[[16, 17]], [[18, 19]]],
            ],
        ),
        (
            # 7 stages, 1 replica
            11,
            {encoder2_template: 1},
            (llm_template_4stages, 1, 2),
            [
                [[[0], [0]]],
                [[[1], [1]]],
                [[[2], [2]]],
                [[[3], [4]]],
                [[[5], [6]]],
                [[[7], [8]]],
                [[[9], [10]]],
            ],
        ),
        (
            # 7 stages, 1 replica
            19,
            {encoder2_template: 1},
            (llm_template_4stages, 2, 2),
            [
                [[[0, 0], [0, 0]]],
                [[[1, 1], [1, 1]]],
                [[[2, 2], [2, 2]]],
                [[[3, 4], [5, 6]]],
                [[[7, 8], [9, 10]]],
                [[[11, 12], [13, 14]]],
                [[[15, 16], [17, 18]]],
            ],
        ),
        (
            # 7 stages, 1 replica
            22,
            {encoder2_template: 2},
            (llm_template_4stages, 2, 2),
            [
                [[[0, 1], [0, 1]]],
                [[[2, 3], [2, 3]]],
                [[[4, 5], [4, 5]]],
                [[[6, 7], [8, 9]]],
                [[[10, 11], [12, 13]]],
                [[[14, 15], [16, 17]]],
                [[[18, 19], [20, 21]]],
            ],
        ),
        (
            # 7 stages, 2 replicas
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            [
                [[[0, 1], [0, 1]], [[2, 3], [2, 3]]],
                [[[4, 5], [4, 5]], [[6, 7], [6, 7]]],
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                [[[16, 17], [18, 19]], [[20, 21], [22, 23]]],
            ],
        ),
    ],
)
def test_init_process_group_mesh_single_encoder(
    world_size: int,
    encoder_template: tuple[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_mesh: list[list[list[list[int]]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultimodalSequentialProcessGroupMesh(
            encoder_templates=encoder_template,
            llm_template=llm_template,
        )
        assert (mesh.mesh == expected_mesh).all()

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_template, llm_template",
    [
        (12, {encoder1_template: 1}, (llm_template_2stages, 1, 1)),
        (8, {encoder1_template: 2}, (llm_template_2stages, 2, 1)),
        (20, {encoder1_template: 1}, (llm_template_4stages, 2, 1)),
        (11, {encoder2_template: 1}, (llm_template_4stages, 1, 2)),
        (19, {encoder2_template: 1}, (llm_template_4stages, 2, 2)),
        (22, {encoder2_template: 2}, (llm_template_4stages, 2, 2)),
        (24, {encoder1_template: 2}, (llm_template_2stages, 2, 2)),
    ],
)
def test_create_group_along_axis_order(
    world_size: int,
    encoder_template: tuple[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    mocker: MockerFixture,
):
    recorded_new_group_calls: dict[int, list] = defaultdict(list)

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

    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultimodalSequentialProcessGroupMesh(
            encoder_templates=encoder_template,
            llm_template=llm_template,
        )
        mesh.get_group_along_axis(mesh.pp_axis)
        mesh.get_group_along_axis(mesh.dp_axis)
        mesh.get_group_along_axis(mesh.tp_axis)
        mesh.get_group_along_axis(mesh.sp_axis)

        dist.destroy_process_group()

    for rank, calls in recorded_new_group_calls.items():
        assert calls == recorded_new_group_calls[0]


@pytest.mark.parametrize(
    "world_size, encoder_template, llm_template, expected_group_ranks",
    [
        (
            12,
            {encoder1_template: 1},
            (llm_template_2stages, 1, 1),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 3, 6, 9),
                    (1, 4, 7, 10),
                    (2, 5, 8, 11),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [
                    (0, 1, 2),
                    (3, 4, 5),
                    (6, 7, 8),
                    (9, 10, 11),
                ],
                MultimodalSequentialProcessGroupMesh.sp_axis: [(i,) for i in range(12)],
                MultimodalSequentialProcessGroupMesh.tp_axis: [(i,) for i in range(12)],
            },
        ),
        (
            8,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 1),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 6),
                    (1, 3, 5, 7),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [(i,) for i in range(8)],
                MultimodalSequentialProcessGroupMesh.sp_axis: [(i,) for i in range(8)],
                MultimodalSequentialProcessGroupMesh.tp_axis: [
                    (i, i + 1) for i in range(0, 8, 2)
                ],
            },
        ),
        (
            20,
            {encoder1_template: 1},
            (llm_template_4stages, 2, 1),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 8, 12, 16),
                    (0, 2, 5, 9, 13, 17),
                    (1, 3, 6, 10, 14, 18),
                    (1, 3, 7, 11, 15, 19),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 6),
                    (5, 7),
                    (8, 10),
                    (9, 11),
                    (12, 14),
                    (13, 15),
                    (16, 18),
                    (17, 19),
                ],
                MultimodalSequentialProcessGroupMesh.sp_axis: [(i,) for i in range(20)],
                MultimodalSequentialProcessGroupMesh.tp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11),
                    (12, 13),
                    (14, 15),
                    (16, 17),
                    (18, 19),
                ],
            },
        ),
        (
            11,
            {encoder2_template: 1},
            (llm_template_4stages, 1, 2),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 1, 2, 3, 5, 7, 9),
                    (0, 1, 2, 4, 6, 8, 10),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [(i,) for i in range(11)],
                MultimodalSequentialProcessGroupMesh.sp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3, 4),
                    (5, 6),
                    (7, 8),
                    (9, 10),
                ],
                MultimodalSequentialProcessGroupMesh.tp_axis: [(i,) for i in range(11)],
            },
        ),
        (
            19,
            {encoder2_template: 1},
            (llm_template_4stages, 2, 2),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 1, 2, 3, 7, 11, 15),
                    (0, 1, 2, 4, 8, 12, 16),
                    (0, 1, 2, 5, 9, 13, 17),
                    (0, 1, 2, 6, 10, 14, 18),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [(i,) for i in range(19)],
                MultimodalSequentialProcessGroupMesh.sp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3, 5),
                    (4, 6),
                    (7, 9),
                    (8, 10),
                    (11, 13),
                    (12, 14),
                    (15, 17),
                    (16, 18),
                ],
                MultimodalSequentialProcessGroupMesh.tp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3, 4),
                    (5, 6),
                    (7, 8),
                    (9, 10),
                    (11, 12),
                    (13, 14),
                    (15, 16),
                    (17, 18),
                ],
            },
        ),
        (
            22,
            {encoder2_template: 2},
            (llm_template_4stages, 2, 2),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 6, 10, 14, 18),
                    (1, 3, 5, 7, 11, 15, 19),
                    (0, 2, 4, 8, 12, 16, 20),
                    (1, 3, 5, 9, 13, 17, 21),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [(i,) for i in range(22)],
                MultimodalSequentialProcessGroupMesh.sp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4,),
                    (5,),
                    (6, 8),
                    (7, 9),
                    (10, 12),
                    (11, 13),
                    (14, 16),
                    (15, 17),
                    (18, 20),
                    (19, 21),
                ],
                MultimodalSequentialProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11),
                    (12, 13),
                    (14, 15),
                    (16, 17),
                    (18, 19),
                    (20, 21),
                ],
            },
        ),
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 2, 2),
            {
                MultimodalSequentialProcessGroupMesh.pp_axis: [
                    (0, 4, 8, 16),
                    (1, 5, 9, 17),
                    (0, 4, 10, 18),
                    (1, 5, 11, 19),
                    (2, 6, 12, 20),
                    (3, 7, 13, 21),
                    (2, 6, 14, 22),
                    (3, 7, 15, 23),
                ],
                MultimodalSequentialProcessGroupMesh.dp_axis: [
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (16, 20),
                    (17, 21),
                    (18, 22),
                    (19, 23),
                ],
                MultimodalSequentialProcessGroupMesh.sp_axis: [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4,),
                    (5,),
                    (6,),
                    (7,),
                    (8, 10),
                    (9, 11),
                    (12, 14),
                    (13, 15),
                    (16, 18),
                    (17, 19),
                    (20, 22),
                    (21, 23),
                ],
                MultimodalSequentialProcessGroupMesh.tp_axis: [
                    (i, i + 1) for i in range(0, 24, 2)
                ],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "axis",
    [
        MultimodalSequentialProcessGroupMesh.pp_axis,
        MultimodalSequentialProcessGroupMesh.dp_axis,
        MultimodalSequentialProcessGroupMesh.tp_axis,
        MultimodalSequentialProcessGroupMesh.sp_axis,
    ],
    ids=["pp", "dp", "tp", "sp"],
)
def test_get_group_along_axis(
    world_size: int,
    encoder_template: tuple[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_group_ranks: dict[int, list[tuple[int, ...]]],
    axis: int,
    mocker: MockerFixture,
):
    if axis not in expected_group_ranks:
        pytest.skip("Axis not in expected_group_ranks")

    recorded_new_group_calls: dict[int, list] = defaultdict(list)

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

    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultimodalSequentialProcessGroupMesh(
            encoder_templates=encoder_template,
            llm_template=llm_template,
        )
        mesh.get_group_along_axis(axis)

        assert list(mesh._ranks_to_group.keys()) == expected_group_ranks[axis]
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, expected_error_type, expected_error_message",
    [
        (
            5,
            {encoder1_template: 1},
            AssertionError,
            "World size 5 is not divisible by num_ranks per replica 4.",
        ),
        (
            7,
            {encoder1_template: 1, encoder2_template: 1},
            AssertionError,
            "All encoder templates must have the same number of stages.",
        ),
        (
            8,
            {encoder1_template: 1, encoder3_template: 2},
            AssertionError,
            "All encoder templates must have the same tensor parallel degree.",
        ),
    ],
)
def test_process_group_mesh_errors(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    expected_error_type: type,
    expected_error_message: str,
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        with pytest.raises(expected_error_type, match=expected_error_message):
            MultimodalSequentialProcessGroupMesh(
                encoder_templates, (llm_template_2stages, 1, 1)
            )

        dist.destroy_process_group()
