import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    ModalProcessGroupMesh,
)


@pytest.mark.parametrize(
    "ranks_per_modal",
    [
        [
            ([0, 1, 2, 3, 12, 13, 14, 15], (2, 2)),
            ([4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23], (2, 4)),
        ],
        [
            ([0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19], (2, 2)),
            (
                [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23],
                (2, 2),
            ),
        ],
        [
            (
                [0, 1, 2, 3, 12, 13, 14, 15],
                (2, 2),
            ),
            (
                [4, 5, 6, 7, 16, 17, 18, 19],
                (2, 2),
            ),
            ([8, 9, 10, 11, 20, 21, 22, 23], (2, 2)),
        ],
    ],
)
def test_get_group(
    ranks_per_modal: list[tuple[list[int], tuple[int, int]]], mocker: MockerFixture
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

    for rank in range(24):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=24
        )

        meshes = []

        for ranks_for_modal, size in ranks_per_modal:
            group = dist.new_group(ranks_for_modal)
            tp_size, pp_size = size
            dp_size = len(ranks_for_modal) // (tp_size * pp_size)
            mesh = ModalProcessGroupMesh(group, pp_size, dp_size, tp_size)
            meshes.append(mesh)

        dist.destroy_process_group()

    for rank, records in recorded_new_group_calls.items():
        # All ranks must call new_group in the same order with the same list of ranks.
        assert len(records) == len(ranks_per_modal)
        assert records == [ranks[0] for ranks in ranks_per_modal]
