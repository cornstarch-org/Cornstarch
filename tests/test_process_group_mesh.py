import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh
from pytest_mock import MockerFixture
import functools
from collections import defaultdict

from transformers import PretrainedConfig


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
            {PipelineTemplate(None, ["0"], [1], [[None, None]]): 1},
            1,
            [[[0], [0]]],
        ],
        [
            {PipelineTemplate(None, ["0"], [1], [[None, None, None]]): 2},
            1,
            [[[0], [0], [0]], [[1], [1], [1]]],
        ],
        [
            {PipelineTemplate(None, ["0"], [1, 1], [[None], [None, None]]): 2},
            1,
            [[[0], [1], [1]], [[2], [3], [3]]],
        ],
        [
            {PipelineTemplate(None, ["0"], [2], [[None, None]]): 1},
            2,
            [[[0, 1], [0, 1]]],
        ],
        [
            {PipelineTemplate(None, ["0"], [4, 4], [[None], [None, None]]): 2},
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
                PipelineTemplate(None, ["0"], [1, 1], [[None], [None, None]]): 1,
                PipelineTemplate(None, ["0"], [1], [[None, None, None]]): 1,
            },
            1,
            [[[0], [1], [1]], [[2], [2], [2]]],
        ],
        [
            {
                PipelineTemplate(None, ["0"], [1, 1], [[None], [None, None]]): 1,
                PipelineTemplate(None, ["0"], [1], [[None, None, None]]): 2,
            },
            1,
            [[[0], [1], [1]], [[2], [2], [2]], [[3], [3], [3]]],
        ],
        [
            {
                PipelineTemplate(None, ["0"], [4, 4], [[None], [None]]): 2,
                PipelineTemplate(None, ["0"], [4], [[None, None]]): 2,
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
                    None, ["0"], [2, 2, 2], [[None], [None], [None, None]]
                ): 2,
                PipelineTemplate(None, ["0"], [2, 2], [[None, None, None], [None]]): 1,
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
                PipelineTemplate(None, ["0"], [2, 2], [[None, None], [None]]): 4,
            },
            2,
            {
                0: [[0, 4, 8, 12], [0, 4, 8, 12]],
                1: [[1, 5, 9, 13], [1, 5, 9, 13]],
                2: [[2, 6, 10, 14]],
                3: [[3, 7, 11, 15]],
                4: [[0, 4, 8, 12], [0, 4, 8, 12]],
                5: [[1, 5, 9, 13], [1, 5, 9, 13]],
                6: [[2, 6, 10, 14]],
                7: [[3, 7, 11, 15]],
                8: [[0, 4, 8, 12], [0, 4, 8, 12]],
                9: [[1, 5, 9, 13], [1, 5, 9, 13]],
                10: [[2, 6, 10, 14]],
                11: [[3, 7, 11, 15]],
                12: [[0, 4, 8, 12], [0, 4, 8, 12]],
                13: [[1, 5, 9, 13], [1, 5, 9, 13]],
                14: [[2, 6, 10, 14]],
                15: [[3, 7, 11, 15]],
            },
        ],
        [
            {
                PipelineTemplate(
                    None, ["0"], [2, 2, 2, 2], [[None], [None], [None, None], [None]]
                ): 1,
                PipelineTemplate(
                    None, ["0"], [2, 2], [[None, None, None], [None, None]]
                ): 2,
            },
            2,
            {
                0: [[0, 8, 12]],
                1: [[1, 9, 13]],
                2: [[2, 8, 12]],
                3: [[3, 9, 13]],
                4: [[4, 8, 12], [4, 10, 14]],
                5: [[5, 9, 13], [5, 11, 15]],
                6: [[6, 10, 14]],
                7: [[7, 11, 15]],
                8: [[0, 8, 12], [2, 8, 12], [4, 8, 12]],
                9: [[1, 9, 13], [3, 9, 13], [5, 9, 13]],
                10: [[4, 10, 14], [6, 10, 14]],
                11: [[5, 11, 15], [7, 11, 15]],
                12: [[0, 8, 12], [2, 8, 12], [4, 8, 12]],
                13: [[1, 9, 13], [3, 9, 13], [5, 9, 13]],
                14: [[4, 10, 14], [6, 10, 14]],
                15: [[5, 11, 15], [7, 11, 15]],
            },
        ],
    ],
)
@pytest.mark.noautofixture
def test_get_dp_groups(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    expected_ranks: dict[int, list[list[int]]],
    mocker: MockerFixture,
):
    recorded_new_group_calls: dict[int, list] = defaultdict(list)

    def record_new_group_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Append ranks to the list so that
            # the list represents the order of creating new groups.
            recorded_new_group_calls[dist.get_rank()].append(args[0])
            return func(*args, **kwargs)

        return wrapper

    mock = mocker.patch(
        "test_process_group_mesh.dist.new_group",
        wraps=record_new_group_call_decorator(dist.new_group),
    )

    for rank in range(16):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=16
        )
        mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        groups = mesh.get_group_along_axis(0)

        if isinstance(groups, dist.ProcessGroup):
            groups = [groups]

        assert len(groups) == len(expected_ranks[rank])
        for group, expected_rank in zip(groups, expected_ranks[rank]):
            ranks = dist.get_process_group_ranks(group)
            assert (
                ranks == expected_rank
            ), f"[rank : {rank}] Expected: {expected_rank}, Got: {ranks}"

        del mesh
        dist.destroy_process_group()

    for rank in range(1, 16):
        assert recorded_new_group_calls[0] == recorded_new_group_calls[rank], (
            f"Expected: {recorded_new_group_calls[0]}, "
            f"Got: {recorded_new_group_calls[rank]}"
        )
