from torch.testing._internal.distributed.fake_pg import FakeStore
import torch.distributed as dist
from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def init_process_group():
    store = FakeStore()
    dist.init_process_group(backend="fake", store=store, rank=0, world_size=4)
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


# def test_heterogeneous_pipelines(
#     pipeline_template: list[PipelineTemplate],
#     num_pipelines: list[int],
#     tp_size: int,
#     expected_mesh: list,
# ):
#     pass
