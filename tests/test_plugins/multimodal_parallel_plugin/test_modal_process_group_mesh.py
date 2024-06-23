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


@pytest.mark.parametrize(
    "world_size, modal_template, execution_order, expected_mesh, expected_ranks",
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
def test_get_group(
    world_size: int,
    modal_template: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_mesh: list[list[list[list[int]]]],
    expected_ranks: dict[PipelineTemplate, list[int]],
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

    meshes = []
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = MultiModalProcessGroupMesh(modal_template, execution_order)
        assert (mesh.mesh == expected_mesh).all()
        assert mesh.modal_to_ranks == expected_ranks
        meshes.append(mesh)
        group = mesh.get_group_along_axis(mesh.dp_axis)

        dist.destroy_process_group()

    for rank, group_calls in recorded_new_group_calls.items():
        assert group_calls == recorded_new_group_calls[0]
