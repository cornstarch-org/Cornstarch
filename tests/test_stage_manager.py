import functools
from collections import defaultdict

import numpy as np
import pytest
import torch.distributed as dist
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.process_group_mesh import HeterogeneousProcessGroupMesh
from cornstarch.stage_manager import HeterogeneousPipelineStageManager
from pytest_mock import MockerFixture
from torch.distributed.distributed_c10d import GroupMember
from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

no_tp_templates = [
    PipelineTemplate("fake", [[None, None], [None, None, None, None]]),
    PipelineTemplate("fake", [[None, None], [None, None, None, None]]),
    PipelineTemplate("fake", [[None], [None, None, None], [None, None]]),
]
no_tp_template_ranks = [
    [[0], [0], [1], [1], [1], [1]],
    [[2], [2], [3], [3], [3], [3]],
    [[4], [5], [5], [5], [6], [6]],
]

tp_templates = [
    PipelineTemplate("fake", [[None, None], [None, None, None, None]]),
    PipelineTemplate("fake", [[None], [None, None, None], [None, None]]),
    PipelineTemplate("fake", [[None], [None, None, None], [None, None]]),
]
tp_template_ranks = [
    [[0, 1], [0, 1], [2, 3], [2, 3], [2, 3], [2, 3]],
    [[4, 5], [6, 7], [6, 7], [6, 7], [8, 9], [8, 9]],
    [[10, 11], [12, 13], [12, 13], [12, 13], [14, 15], [14, 15]],
]


@pytest.mark.parametrize(
    "pipeline_templates, tp_size, world_size",
    [
        [
            no_tp_templates,
            1,
            7,
        ],
        [
            tp_templates,
            2,
            16,
        ],
    ],
)
def test_stage_manager_process_group_init_order_matches(
    pipeline_templates: list[PipelineTemplate],
    tp_size: int,
    world_size: int,
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
        "test_stage_manager.dist.new_group",
        wraps=record_new_group_call_decorator(dist.new_group),
    )

    pp_axis = 1
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, pp_axis)
        del pg_mesh
        del stage_manager
        dist.destroy_process_group()

    for rank in range(1, world_size):
        assert (
            recorded_new_group_calls[0] == recorded_new_group_calls[rank]
        ), f"new_group calls are not in the same order for rank {rank}"


class TestStageManagerClass(MultiThreadedTestCase):
    pp_axis = 1

    @property
    def world_size(self):
        return 16

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @parametrize(
        "pipeline_templates, tp_size, world_size, ranks_to_expected_num_stages",
        [
            [
                no_tp_templates,
                1,
                7,
                {
                    (0, 1, 2, 3): 2,
                    (4, 5, 6): 3,
                },
            ],
            [
                tp_templates,
                2,
                16,
                {
                    (0, 1, 2, 3): 2,
                    (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15): 3,
                },
            ],
        ],
        name_fn=lambda templates, tp_size, world_size, _: (
            "no_tp_templates" if tp_size == 1 else "tp_templates",
        ),
    )
    def test_num_stages(
        self,
        pipeline_templates: list[PipelineTemplate],
        tp_size: int,
        world_size: int,
        ranks_to_expected_num_stages: dict[tuple[int, ...], int],
    ):
        if self.rank >= world_size:
            return

        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, self.pp_axis)

        assert stage_manager.num_stages == next(
            (v for k, v in ranks_to_expected_num_stages.items() if self.rank in k)
        )

    @parametrize(
        "pipeline_templates, tp_size, world_size, layers_to_expected_ranks_in_group",
        [
            [
                no_tp_templates,
                1,
                7,
                {
                    (0, 1): [4, 5],
                    (0, 2): [0, 1, 2, 3, 4, 5],
                    (0, 5): [0, 1, 2, 3, 4, 6],
                    (1, 2): [0, 1, 2, 3],
                    (1, 4): [0, 1, 2, 3, 5, 6],
                },
            ],
            [
                tp_templates,
                2,
                16,
                {
                    (0, 1): [4, 5, 6, 7, 10, 11, 12, 13],
                    (0, 5): [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15],
                    (0, 3): [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
                    (1, 2): [0, 1, 2, 3],
                    (3, 5): [6, 8, 7, 9, 12, 14, 13, 15],
                },
            ],
        ],
        name_fn=lambda templates, tp_size, world_size, _: (
            "no_tp_templates" if tp_size == 1 else "tp_templates",
        ),
    )
    def test_init_process_group_by_layers(
        self,
        pipeline_templates: list[PipelineTemplate],
        tp_size: int,
        world_size: int,
        layers_to_expected_ranks_in_group: dict[tuple[int, ...], list[int]],
    ):
        if self.rank >= world_size:
            return

        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, self.pp_axis)

        for layers, expected_ranks in layers_to_expected_ranks_in_group.items():
            group = stage_manager.init_process_group_by_layers(layers)
            if self.rank in expected_ranks:
                assert group is not None
                assert self.rank in dist.get_process_group_ranks(group)
            elif dist.get_world_size(group) == 1:
                assert dist.get_process_group_ranks(group)[0] == self.rank
            else:
                assert group is None or group == GroupMember.NON_GROUP_MEMBER

    @parametrize(
        "pipeline_templates, tp_size, world_size, rank_to_expected_prev_next_rank",
        [
            [
                no_tp_templates,
                1,
                7,
                {
                    0: (1, 1),
                    1: (0, 0),
                    2: (3, 3),
                    3: (2, 2),
                    4: (6, 5),
                    5: (4, 6),
                    6: (5, 4),
                },
            ],
            [
                tp_templates,
                2,
                16,
                {
                    0: (2, 2),
                    1: (3, 3),
                    2: (0, 0),
                    3: (1, 1),
                    4: (8, 6),
                    5: (9, 7),
                    6: (4, 8),
                    7: (5, 9),
                    8: (6, 4),
                    9: (7, 5),
                    10: (14, 12),
                    11: (15, 13),
                    12: (10, 14),
                    13: (11, 15),
                    14: (12, 10),
                    15: (13, 11),
                },
            ],
        ],
        name_fn=lambda templates, tp_size, world_size, _: (
            "no_tp_templates" if tp_size == 1 else "tp_templates",
        ),
    )
    def test_prev_next_ranks(
        self,
        pipeline_templates: list[PipelineTemplate],
        tp_size: int,
        world_size: int,
        rank_to_expected_prev_next_rank: dict[int, tuple[int, int]],
    ):
        if self.rank >= world_size:
            return

        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, self.pp_axis)

        assert stage_manager.prev_rank == rank_to_expected_prev_next_rank[self.rank][0]
        assert stage_manager.next_rank == rank_to_expected_prev_next_rank[self.rank][1]

    @parametrize(
        "pipeline_templates, tp_size, world_size, ranks_to_expected_p2p_ranks",
        [
            [
                no_tp_templates,
                1,
                7,
                {
                    0: [(0, 1)],
                    1: [(0, 1)],
                    2: [(2, 3)],
                    3: [(2, 3)],
                    4: [(4, 5)],
                    5: [(4, 5), (5, 6)],
                    6: [(5, 6)],
                },
            ],
            [
                tp_templates,
                2,
                16,
                {
                    0: [(0, 2)],
                    1: [(1, 3)],
                    2: [(0, 2)],
                    3: [(1, 3)],
                    4: [(4, 6)],
                    5: [(5, 7)],
                    6: [(4, 6), (6, 8)],
                    7: [(5, 7), (7, 9)],
                    8: [(6, 8)],
                    9: [(7, 9)],
                    10: [(10, 12)],
                    11: [(11, 13)],
                    12: [(10, 12), (12, 14)],
                    13: [(11, 13), (13, 15)],
                    14: [(12, 14)],
                    15: [(13, 15)],
                },
            ],
        ],
        name_fn=lambda templates, tp_size, world_size, _: (
            "no_tp_templates" if tp_size == 1 else "tp_templates",
        ),
    )
    def test_ranks_in_p2p_groups(
        self, pipeline_templates, tp_size, world_size, ranks_to_expected_p2p_ranks
    ):
        if self.rank >= world_size:
            return

        pg_mesh = HeterogeneousProcessGroupMesh(pipeline_templates, tp_size)
        stage_manager = HeterogeneousPipelineStageManager(pg_mesh, self.pp_axis)

        assert np.array_equal(
            list(stage_manager.p2p_groups.keys()),
            ranks_to_expected_p2p_ranks[self.rank],
        ), f"rank {self.rank} has wrong p2p groups"


instantiate_parametrized_tests(TestStageManagerClass)
