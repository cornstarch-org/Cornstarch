from torch.testing._internal.distributed.fake_pg import FakeStore
import torch.distributed as dist
from torch.distributed.distributed_c10d import GroupMember
from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh
from pipeline_template.stage_manager import HeterogeneousPipelineStageManager

from unittest import mock
import pytest
from pytest_mock import MockerFixture
import numpy as np
from collections import defaultdict
import gc
import functools


def init_process_group(rank: int):
    store = FakeStore()
    dist.init_process_group(backend="fake", store=store, rank=rank, world_size=16)


@pytest.fixture(autouse=True)
def destroy_process_group():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


no_tp_templates = {
    PipelineTemplate(
        ["0"],
        [1, 1],
        [[None, None], [None, None, None, None]],
    ): 2,
    PipelineTemplate(
        ["0"],
        [1, 1, 1],
        [[None], [None, None, None], [None, None]],
    ): 1,
}
no_tp_template_ranks = [
    [[0], [0], [1], [1], [1], [1]],
    [[2], [2], [3], [3], [3], [3]],
    [[4], [5], [5], [5], [6], [6]],
]

tp_templates = {
    PipelineTemplate(
        ["0"],
        [2, 2],
        [[None, None], [None, None, None, None]],
    ): 1,
    PipelineTemplate(
        ["0"],
        [2, 2, 2],
        [[None], [None, None, None], [None, None]],
    ): 2,
}
tp_template_ranks = [
    [[0, 1], [0, 1], [2, 3], [2, 3], [2, 3], [2, 3]],
    [[4, 5], [6, 7], [6, 7], [6, 7], [8, 9], [8, 9]],
    [[10, 11], [12, 13], [12, 13], [12, 13], [14, 15], [14, 15]],
]


# def test_colossal_stage_manager_group_call_order_match(
#     mocker: MockerFixture,
# ):
#     from colossalai.pipeline import PipelineStageManager
#     from colossalai.cluster import ProcessGroupMesh

#     recorded_new_group_calls: dict[int, list[list[int]]] = defaultdict(list)

#     def record_new_group_call(ranks: int, *args, **kwargs):
#         # Append ranks to the list so that
#         # the list represents the order of creating new groups.
#         recorded_new_group_calls[dist.get_rank()].append(ranks)

#     mock = mocker.patch(
#         "test_stage_manager.dist.new_group", side_effect=record_new_group_call
#     )
#     pp_axis = 1
#     for rank in range(16):
#         init_process_group(rank)
#         pg_mesh = ProcessGroupMesh(4, 2, 2)
#         stage_manager = PipelineStageManager(pg_mesh, pp_axis)
#         del pg_mesh
#         del stage_manager
#         gc.collect()

#     # The order of calling new_group must be the same across all ranks
#     for rank in range(16):
#         assert recorded_new_group_calls[0] == recorded_new_group_calls[rank]


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, ranks",
    [
        [no_tp_templates, 1, list(range(7))],
        [tp_templates, 2, list(range(16))],
    ],
)
def test_stage_manager_group_call_order_match(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    ranks: list[int],
    mocker: MockerFixture,
):
    recorded_new_group_calls: dict[int, list[list[int]]] = defaultdict(list)

    def record_new_group_call_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Append ranks to the list so that
            # the list represents the order of creating new groups.
            recorded_new_group_calls[dist.get_rank()].append(args[0])
            return func(*args, **kwargs)

        return wrapper

    mock = mocker.patch(
        "test_stage_manager.dist.new_group",
        wraps=record_new_group_call_decorator(dist.new_group),
    )
    pp_axis = 1
    for rank in ranks:
        init_process_group(rank)
        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)
        del pg_mesh
        del stage_manager
        dist.destroy_process_group()
        gc.collect()

    # The order of calling new_group must be the same across all ranks
    for rank in ranks:
        assert (
            recorded_new_group_calls[ranks[0]] == recorded_new_group_calls[rank]
        ), f"new_group calls are not in the same order for rank {rank}"


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, ranks, expected_num_stages, expected_stages",
    [
        [no_tp_templates, 1, [0, 1], 2, [0, 1]],
        [no_tp_templates, 1, [2, 3], 2, [0, 1]],
        [no_tp_templates, 1, [4, 5, 6], 3, [0, 1, 2]],
        [tp_templates, 2, [0, 2], 2, [0, 1]],
        [tp_templates, 2, [1, 3], 2, [0, 1]],
        [tp_templates, 2, [4, 6, 8], 3, [0, 1, 2]],
        [tp_templates, 2, [5, 7, 9], 3, [0, 1, 2]],
        [tp_templates, 2, [10, 12, 14], 3, [0, 1, 2]],
        [tp_templates, 2, [11, 13, 15], 3, [0, 1, 2]],
    ],
    ids=[f"no_tp_pipeline{i}" for i in range(3)]
    + [f"tp_pipeline{i}" for i in range(3 * 2)],
)
def test_pipeline_num_stages(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    ranks: list[int],
    expected_num_stages: int,
    expected_stages: list[int],
):
    for rank, expected_stage in zip(ranks, expected_stages):
        init_process_group(rank)
        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, 1)
        assert stage_manager.num_stages == expected_num_stages
        assert stage_manager.stage == expected_stage
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, rank, p2p_ranks",
    [
        [no_tp_templates, 1, 0, [(0, 1)]],
        [no_tp_templates, 1, 1, [(0, 1)]],
        [no_tp_templates, 1, 2, [(2, 3)]],
        [no_tp_templates, 1, 3, [(2, 3)]],
        [no_tp_templates, 1, 4, [(4, 5)]],
        [no_tp_templates, 1, 5, [(4, 5), (5, 6)]],
        [no_tp_templates, 1, 6, [(5, 6)]],
        [tp_templates, 2, 0, [(0, 2)]],
        [tp_templates, 2, 1, [(1, 3)]],
        [tp_templates, 2, 2, [(0, 2)]],
        [tp_templates, 2, 3, [(1, 3)]],
        [tp_templates, 2, 4, [(4, 6)]],
        [tp_templates, 2, 5, [(5, 7)]],
        [tp_templates, 2, 6, [(4, 6), (6, 8)]],
        [tp_templates, 2, 7, [(5, 7), (7, 9)]],
        [tp_templates, 2, 8, [(6, 8)]],
        [tp_templates, 2, 9, [(7, 9)]],
        [tp_templates, 2, 10, [(10, 12)]],
        [tp_templates, 2, 11, [(11, 13)]],
        [tp_templates, 2, 12, [(10, 12), (12, 14)]],
        [tp_templates, 2, 13, [(11, 13), (13, 15)]],
        [tp_templates, 2, 14, [(12, 14)]],
        [tp_templates, 2, 15, [(13, 15)]],
    ],
    ids=[f"no_tp_rank{i}" for i in range(7)] + [f"tp_rank{i}" for i in range(16)],
)
def test_ranks_in_p2p_groups(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    rank: int,
    p2p_ranks: list[tuple[int, int]],
):
    pp_axis = 1
    init_process_group(rank)
    pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
    stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)
    assert np.array_equal(list(stage_manager.p2p_groups.keys()), p2p_ranks)


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, ranks, layers",
    [
        [no_tp_templates, 1, [4, 5], [0, 1]],
        [no_tp_templates, 1, [0, 1, 2, 3, 4, 5], [0, 2]],
        [no_tp_templates, 1, [0, 1, 2, 3, 4, 6], [0, 5]],
        [no_tp_templates, 1, [0, 1, 2, 3], [1, 2]],
        [no_tp_templates, 1, [0, 1, 2, 3, 5, 6], [1, 4]],
        [no_tp_templates, 1, [], [2, 3]],
    ],
)
def test_init_process_group_by_layers(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    ranks: list[int],
    layers: list[int],
):
    pp_axis = 1
    for rank in ranks:
        init_process_group(rank)
        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)

        group = stage_manager.init_process_group_by_layers(layers)
        if rank in ranks:
            assert group is not None
            assert rank in dist.get_process_group_ranks(group)
        else:
            assert group is None

        del pg_mesh
        del stage_manager
        dist.destroy_process_group()
        gc.collect()


# @pytest.mark.parametrize(
#     "pipeline_templates, tp_size, ranks, layers, expected_group_ranks",
#     [
#         [no_tp_templates, 1, [0, 1], [0, 4], [0, 1]],
#         [no_tp_templates, 1, [0, 1], [0, 1], None],
#         [no_tp_templates, 1, [2, 3], [0, 1], None],
#         [no_tp_templates, 1, [2, 3], [3, 4], None],
#         [no_tp_templates, 1, [2, 3], [0, 5], [2, 3]],
#         [no_tp_templates, 1, [4, 5, 6], [0, 1], [4, 5]],
#         [no_tp_templates, 1, [4, 5, 6], [0, 5], [4, 6]],
#         [no_tp_templates, 1, [4, 5, 6], [3, 4], [5, 6]],
#         # [tp_templates, 2, 16, [0, 1], []],
#         # [tp_templates, 2, 16, [0, 1], []],
#         # [tp_templates, 2, 16, [0, 1], []],
#         # [tp_templates, 2, 16, [0, 2], []],
#         # [tp_templates, 2, 16, [1, 2], []],
#         # [tp_templates, 2, 16, [0, 1], []],
#         # [tp_templates, 2, 16, [0, 2], []],
#         # [tp_templates, 2, 16, [1, 2], []],
#     ],
# )
# def test_init_process_group_by_layers(
#     pipeline_templates: dict[PipelineTemplate, int],
#     tp_size: int,
#     ranks: list[int],
#     layers: list[int],
#     expected_group_ranks: list[int] | None,
# ):
#     pp_axis = 1
#     for rank in ranks:
#         init_process_group(rank)
#         pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
#         stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)

#         group = stage_manager.init_process_group_by_layers(layers)
#         if expected_group_ranks is None or rank not in expected_group_ranks:
#             assert group is None
#         else:
#             assert expected_group_ranks == dist.get_process_group_ranks(group)

#         del pg_mesh
#         del stage_manager
#         dist.destroy_process_group()
#         gc.collect()


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, rank, expected_prev_next_ranks",
    [
        [no_tp_templates, 1, 0, (1, 1)],
        [no_tp_templates, 1, 1, (0, 0)],
        [no_tp_templates, 1, 2, (3, 3)],
        [no_tp_templates, 1, 3, (2, 2)],
        [no_tp_templates, 1, 4, (6, 5)],
        [no_tp_templates, 1, 5, (4, 6)],
        [no_tp_templates, 1, 6, (5, 4)],
        [tp_templates, 2, 0, (2, 2)],
        [tp_templates, 2, 1, (3, 3)],
        [tp_templates, 2, 2, (0, 0)],
        [tp_templates, 2, 3, (1, 1)],
        [tp_templates, 2, 4, (8, 6)],
        [tp_templates, 2, 5, (9, 7)],
        [tp_templates, 2, 6, (4, 8)],
        [tp_templates, 2, 7, (5, 9)],
        [tp_templates, 2, 8, (6, 4)],
        [tp_templates, 2, 9, (7, 5)],
        [tp_templates, 2, 10, (14, 12)],
        [tp_templates, 2, 11, (15, 13)],
        [tp_templates, 2, 12, (10, 14)],
        [tp_templates, 2, 13, (11, 15)],
        [tp_templates, 2, 14, (12, 10)],
        [tp_templates, 2, 15, (13, 11)],
    ],
    ids=[f"no_tp_rank{i}" for i in range(7)] + [f"tp_rank{i}" for i in range(16)],
)
def test_prev_next_ranks(
    pipeline_templates: dict[PipelineTemplate, int],
    tp_size: int,
    rank: int,
    expected_prev_next_ranks: tuple[int, int],
):
    pp_axis = 1
    init_process_group(rank)
    pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
    stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)
    assert stage_manager.prev_rank == expected_prev_next_ranks[0]
    assert stage_manager.next_rank == expected_prev_next_ranks[1]
