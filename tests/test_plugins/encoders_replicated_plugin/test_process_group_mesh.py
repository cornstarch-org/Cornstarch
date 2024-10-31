import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
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
    "world_size, llm_template, expected_mesh",
    [
        (
            # 2 stages, 2 replicas
            4,
            (llm_template_2stages, 1, 1),
            [
                [[[0]], [[1]]],
                [[[2]], [[3]]],
            ],
        ),
        (
            # 2 stages, 2 replicas
            16,
            (llm_template_2stages, 2, 2),
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            ],
        ),
        (
            # 4 stages, 3 replicas
            24,
            (llm_template_4stages, 2, 1),
            [
                [[[0, 1]], [[2, 3]], [[4, 5]]],
                [[[6, 7]], [[8, 9]], [[10, 11]]],
                [[[12, 13]], [[14, 15]], [[16, 17]]],
                [[[18, 19]], [[20, 21]], [[22, 23]]],
            ],
        ),
        (
            # 4 stages, 2 replicas
            32,
            (llm_template_4stages, 1, 4),
            [
                [[[0], [1], [2], [3]], [[4], [5], [6], [7]]],
                [[[8], [9], [10], [11]], [[12], [13], [14], [15]]],
                [[[16], [17], [18], [19]], [[20], [21], [22], [23]]],
                [[[24], [25], [26], [27]], [[28], [29], [30], [31]]],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "encoder_templates",
    [
        ([encoder1_template]),
        ([encoder3_template]),
        ([encoder1_template, encoder3_template]),
    ],
)
def test_init_process_group_mesh(
    world_size: int,
    encoder_templates: list[PipelineTemplate],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_mesh: list[list[list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = EncodersReplicatedProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=llm_template,
        )
        assert (mesh.mesh == expected_mesh).all()

        dist.destroy_process_group()


@pytest.mark.parametrize("llm_template", [llm_template_2stages, llm_template_4stages])
@pytest.mark.parametrize("tp_size, sp_size", [(1, 1), (4, 1), (2, 2), (1, 4)])
@pytest.mark.parametrize(
    "encoder_templates",
    [
        ([encoder1_template]),
        ([encoder3_template]),
        ([encoder1_template, encoder3_template]),
    ],
)
def test_create_group_along_axis_order(
    encoder_templates: list[PipelineTemplate],
    llm_template: PipelineTemplate,
    tp_size: int,
    sp_size: int,
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

    world_size = 2 * tp_size * sp_size * llm_template.num_stages
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = EncodersReplicatedProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=(llm_template, tp_size, sp_size),
        )
        mesh.get_group_along_axis(mesh.pp_axis)
        mesh.get_group_along_axis(mesh.dp_axis)
        mesh.get_group_along_axis(mesh.tp_axis)
        mesh.get_group_along_axis(mesh.sp_axis)

        dist.destroy_process_group()

    for rank, calls in recorded_new_group_calls.items():
        assert calls == recorded_new_group_calls[0]


@pytest.mark.parametrize(
    "llm_template, expected_group_ranks",
    [
        (
            # 2 stages, 2 ranks per replica
            (llm_template_2stages, 1, 1),
            {
                EncodersReplicatedProcessGroupMesh.pp_axis: [(0, 2), (1, 3)],
                EncodersReplicatedProcessGroupMesh.dp_axis: [(0, 1), (2, 3)],
                EncodersReplicatedProcessGroupMesh.tp_axis: [(0,), (1,), (2,), (3,)],
                EncodersReplicatedProcessGroupMesh.sp_axis: [(0,), (1,), (2,), (3,)],
            },
        ),
        (
            # 2 stages, 8 ranks per replica
            (llm_template_2stages, 2, 2),
            {
                EncodersReplicatedProcessGroupMesh.pp_axis: [
                    (0, 8),
                    (1, 9),
                    (2, 10),
                    (3, 11),
                    (4, 12),
                    (5, 13),
                    (6, 14),
                    (7, 15),
                ],
                EncodersReplicatedProcessGroupMesh.dp_axis: [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                ],
                EncodersReplicatedProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11),
                    (12, 13),
                    (14, 15),
                ],
                EncodersReplicatedProcessGroupMesh.sp_axis: [
                    (0, 2),
                    (1, 3),
                    (4, 6),
                    (5, 7),
                    (8, 10),
                    (9, 11),
                    (12, 14),
                    (13, 15),
                ],
            },
        ),
        (
            # 4 stages, 16 ranks per replica
            (llm_template_4stages, 4, 1),
            {
                EncodersReplicatedProcessGroupMesh.pp_axis: [
                    (0, 8, 16, 24),
                    (1, 9, 17, 25),
                    (2, 10, 18, 26),
                    (3, 11, 19, 27),
                    (4, 12, 20, 28),
                    (5, 13, 21, 29),
                    (6, 14, 22, 30),
                    (7, 15, 23, 31),
                ],
                EncodersReplicatedProcessGroupMesh.dp_axis: [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (16, 20),
                    (17, 21),
                    (18, 22),
                    (19, 23),
                    (24, 28),
                    (25, 29),
                    (26, 30),
                    (27, 31),
                ],
                EncodersReplicatedProcessGroupMesh.tp_axis: [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                    (16, 17, 18, 19),
                    (20, 21, 22, 23),
                    (24, 25, 26, 27),
                    (28, 29, 30, 31),
                ],
                EncodersReplicatedProcessGroupMesh.sp_axis: [(i,) for i in range(32)],
            },
        ),
        (
            # 4 stages, 16 ranks per replica
            (llm_template_4stages, 1, 4),
            {
                EncodersReplicatedProcessGroupMesh.pp_axis: [
                    (0, 8, 16, 24),
                    (1, 9, 17, 25),
                    (2, 10, 18, 26),
                    (3, 11, 19, 27),
                    (4, 12, 20, 28),
                    (5, 13, 21, 29),
                    (6, 14, 22, 30),
                    (7, 15, 23, 31),
                ],
                EncodersReplicatedProcessGroupMesh.dp_axis: [
                    (0, 4),
                    (1, 5),
                    (2, 6),
                    (3, 7),
                    (8, 12),
                    (9, 13),
                    (10, 14),
                    (11, 15),
                    (16, 20),
                    (17, 21),
                    (18, 22),
                    (19, 23),
                    (24, 28),
                    (25, 29),
                    (26, 30),
                    (27, 31),
                ],
                EncodersReplicatedProcessGroupMesh.tp_axis: [(i,) for i in range(32)],
                EncodersReplicatedProcessGroupMesh.sp_axis: [
                    (0, 1, 2, 3),
                    (4, 5, 6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                    (16, 17, 18, 19),
                    (20, 21, 22, 23),
                    (24, 25, 26, 27),
                    (28, 29, 30, 31),
                ],
            },
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
@pytest.mark.parametrize(
    "axis",
    [
        EncodersReplicatedProcessGroupMesh.pp_axis,
        EncodersReplicatedProcessGroupMesh.dp_axis,
        EncodersReplicatedProcessGroupMesh.tp_axis,
        EncodersReplicatedProcessGroupMesh.sp_axis,
    ],
    ids=["pp", "dp", "tp", "sp"],
)
def test_get_group_along_axis(
    encoder_templates: list[PipelineTemplate],
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

    tp_size, sp_size = llm_template[1], llm_template[2]
    world_size = 2 * tp_size * sp_size * llm_template[0].num_stages
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = EncodersReplicatedProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=llm_template,
        )
        mesh.get_group_along_axis(axis)

        assert list(mesh._ranks_to_group.keys()) == expected_group_ranks[axis]
        dist.destroy_process_group()
