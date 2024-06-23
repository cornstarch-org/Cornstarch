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

encoder1_template = PipelineTemplate(
    "encoder1", [["layer.0", "layer.1"], ["layer.2", "layer.3"]]
)
encoder2_template = PipelineTemplate(
    "encoder2", [["layer.0", "layer.1"], ["layer.2", "layer.3"], ["layer.4", "layer.5"]]
)

llm_template_2stages = PipelineTemplate(
    "llm", [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4", "layer.5"]]
)
llm_template_4stages = PipelineTemplate(
    "llm",
    [
        ["layer.0", "layer.1"],
        ["layer.2", "layer.3"],
        ["layer.4", "layer.5"],
        ["layer.6", "layer.7"],
    ],
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, modal_templates, execution_order, expected_mesh, expected_ranks",
    [
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
            [
                [
                    [0, 0, 1, 1],
                    [2, 2, 3, 3],
                ],
                [
                    [4, 4, 5, 5],
                    [6, 6, 7, 7],
                ],
                [
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ],
                [
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                ],
            ],
            {
                encoder1_template: list(range(0, 8)),
                llm_template_2stages: list(range(8, 24)),
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2, llm_template_2stages: 4},
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
            [
                [
                    [0, 0, 1, 1],
                ],
                [
                    [2, 2, 3, 3],
                ],
                [
                    [4, 4, 5, 5],
                ],
                [
                    [6, 6, 7, 7],
                ],
                [
                    [8, 8, 9, 9],
                ],
                [
                    [10, 11, 12, 13],
                ],
                [
                    [14, 15, 16, 17],
                ],
            ],
            {
                encoder1_template: list(range(0, 4)),
                encoder2_template: list(range(4, 10)),
                llm_template_2stages: list(range(10, 18)),
            },
        ),
        (
            84,
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
            [
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                ],
                [
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                    [20, 21, 22, 23],
                ],
                [
                    [24, 25, 26, 27],
                    [28, 29, 30, 31],
                    [32, 33, 34, 35],
                ],
                [
                    [36, 37, 38, 39],
                    [40, 41, 42, 43],
                    [44, 45, 46, 47],
                ],
                [
                    [48, 49, 50, 51],
                    [52, 53, 54, 55],
                    [56, 57, 58, 59],
                ],
                [
                    [60, 61, 62, 63],
                    [64, 65, 66, 67],
                    [68, 69, 70, 71],
                ],
                [
                    [72, 73, 74, 75],
                    [76, 77, 78, 79],
                    [80, 81, 82, 83],
                ],
            ],
            {
                encoder2_template: list(range(0, 36)),
                llm_template_4stages: list(range(36, 84)),
            },
        ),
    ],
)
def test_init_process_group_mesh(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_mesh: list[list[list[list[int]]]],
    expected_ranks: dict[PipelineTemplate, list[int]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        assert (mesh.mesh == expected_mesh).all()
        assert mesh.modal_to_ranks == expected_ranks

        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, modal_templates, execution_order, expected_group_ranks",
    (
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 4, 8, 16),
                    (0, 4, 9, 17),
                    (1, 5, 10, 18),
                    (1, 5, 11, 19),
                    (2, 6, 12, 20),
                    (2, 6, 13, 21),
                    (3, 7, 14, 22),
                    (3, 7, 15, 23),
                ],
                MultiModalProcessGroupMesh.dp_axis: [
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
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9, 10, 11),
                    (12, 13, 14, 15),
                    (16, 17, 18, 19),
                    (20, 21, 22, 23),
                ],
            },
        ),
        (
            18,
            {encoder1_template: 2, encoder2_template: 2, llm_template_2stages: 4},
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
            {
                MultiModalProcessGroupMesh.pp_axis: [
                    (0, 2, 4, 6, 8, 10, 14),
                    (0, 2, 4, 6, 8, 11, 15),
                    (1, 3, 5, 7, 9, 12, 16),
                    (1, 3, 5, 7, 9, 13, 17),
                ],
                MultiModalProcessGroupMesh.dp_axis: [(i,) for i in range(18)],
                MultiModalProcessGroupMesh.tp_axis: [
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11, 12, 13),
                    (14, 15, 16, 17),
                ],
            },
        ),
        (
            84,
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
            {
                # (0, 12, 24, 36, 48, 60, 72), (1, 13, 25, 37, 49, 61, 73), ...
                MultiModalProcessGroupMesh.pp_axis: [
                    tuple(range(i, i + 12 * 7, 12)) for i in range(12)
                ],
                # (0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11), (12, 16, 20), (13, 17, 21), ...
                MultiModalProcessGroupMesh.dp_axis: [
                    tuple([i, i + 4, i + 8])
                    for j in range(0, 84, 12)
                    for i in range(j, j + 4)
                ],
                # (0, 1, 2, 3), (4, 5, 6, 7), ...
                MultiModalProcessGroupMesh.tp_axis: [
                    tuple(range(i, i + 4)) for i in range(0, 84, 4)
                ],
            },
        ),
    ),
)
@pytest.mark.parametrize(
    "axis",
    [
        MultiModalProcessGroupMesh.pp_axis,
        MultiModalProcessGroupMesh.dp_axis,
        MultiModalProcessGroupMesh.tp_axis,
    ],
)
def test_get_group_along_axis(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_group_ranks: dict[int, list[tuple[int, ...]]],
    axis: int,
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

        mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        mesh.get_group_along_axis(axis)
        assert list(mesh._ranks_to_group.keys()) == expected_group_ranks[axis]
        dist.destroy_process_group()

    for rank, calls in recorded_new_group_calls.items():
        assert calls == recorded_new_group_calls[0]
