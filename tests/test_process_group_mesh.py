import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh


@pytest.fixture(autouse=True)
def init_process_group(request: pytest.FixtureRequest):
    if "noautofixture" in request.keywords:
        yield
    else:
        store = FakeStore()
        dist.init_process_group(backend="fake", store=store, rank=0, world_size=8)
        yield
        dist.destroy_process_group()


# Simulating 4-stage pipeline with different number of modules per stage.
modules_per_stage: list[list[torch.nn.Module]] = [
    [None, None],
    [None, None, None],
    [None, None, None, None],
    [None],
]


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, expected_mesh",
    [
        [
            {
                PipelineTemplate(
                    ["0"],
                    [1],
                    [[None, None]],
                ): 1
            },
            1,
            [[[0], [0]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [1],
                    [[None, None, None]],
                ): 2
            },
            1,
            [[[0], [0], [0]], [[1], [1], [1]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [1, 1],
                    [[None], [None, None]],
                ): 2
            },
            1,
            [[[0], [1], [1]], [[2], [3], [3]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [2],
                    [[None, None]],
                ): 1
            },
            2,
            [[[0, 1], [0, 1]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [4, 4],
                    [[None], [None, None]],
                ): 2
            },
            4,
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7]],
                [[8, 9, 10, 11], [12, 13, 14, 15], [12, 13, 14, 15]],
            ],
        ],
    ],
)
def test_homogeneous_pipelines(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    expected_mesh: list,
):
    mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
    np.array_equal(mesh._mesh, expected_mesh)


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, expected_mesh",
    [
        [
            {
                PipelineTemplate(
                    ["0"],
                    [1, 1],
                    [[None], [None, None]],
                ): 1,
                PipelineTemplate(
                    ["0"],
                    [1],
                    [[None, None, None]],
                ): 1,
            },
            1,
            [[[0], [1], [1]], [[2], [2], [2]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [1, 1],
                    [[None], [None, None]],
                ): 1,
                PipelineTemplate(
                    ["0"],
                    [1],
                    [[None, None, None]],
                ): 2,
            },
            1,
            [[[0], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [4, 4],
                    [[None], [None]],
                ): 2,
                PipelineTemplate(
                    ["0"],
                    [4],
                    [[None, None]],
                ): 2,
            },
            4,
            [
                [[0, 1, 2, 3], [4, 5, 6, 7]],
                [[8, 9, 10, 11], [12, 13, 14, 15]],
                [[16, 17, 18, 19], [16, 17, 18, 29]],
                [[20, 21, 22, 23], [20, 21, 22, 23]],
            ],
        ],
        [
            {
                PipelineTemplate(
                    ["0"],
                    [2, 2, 2],
                    [[None], [None], [None, None]],
                ): 2,
                PipelineTemplate(
                    ["0"],
                    [2, 2],
                    [[None, None, None], [None]],
                ): 1,
            },
            2,
            [
                [[0, 1], [2, 3], [4, 5], [4, 5]],
                [[6, 7], [8, 9], [10, 11], [10, 11]],
                [[12, 13], [12, 13], [12, 13], [14, 15]],
                [[16, 17], [16, 17], [16, 17], [18, 19]],
            ],
        ],
    ],
)
def test_heterogeneous_pipelines(
    pipeline_templates: dict[PipelineTemplate, int], tp_size: int, expected_mesh: list
):
    assert all(
        all(num_gpus == tp_size for num_gpus in template.gpus_per_stage)
        for template in pipeline_templates.keys()
    ), "Heterogeneous tensor parallel stage is not supported yet."
    mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
    np.array_equal(mesh._mesh, expected_mesh)


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, expected_ranks",
    [
        [
            {
                PipelineTemplate(["0"], [2, 2], [[None, None], [None]]): 4,
            },
            2,
            [
                [0, 4, 8, 12],
                [1, 5, 9, 13],
                [2, 6, 10, 14],
                [3, 7, 11, 15],
            ],
        ],
        [
            {
                PipelineTemplate(
                    ["0"], [2, 2, 2, 2], [[None], [None], [None, None], [None]]
                ): 1,
                PipelineTemplate(["0"], [2, 2], [[None, None, None], [None, None]]): 2,
            },
            2,
            [
                [0, 8, 12],
                [1, 9, 13],
                [2, 8, 12],
                [3, 9, 13],
                [4, 8, 12],
                [5, 9, 13],
                [4, 10, 14],
                [5, 11, 15],
                [6, 10, 14],
                [7, 11, 15],
            ],
        ],
    ],
)
@pytest.mark.noautofixture
def test_get_dp_groups(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    expected_ranks: list[list[int]],
):
    for rank in range(8):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=16
        )
        mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        groups = mesh.get_group_along_axis(0)

        for group in groups:
            ranks = dist.get_process_group_ranks(group)
            assert ranks in expected_ranks

        dist.destroy_process_group()
