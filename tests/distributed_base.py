import os
import random
import sys
from typing import Optional
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA


class GlooDistributedTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        raise NotImplementedError

    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe, **kwargs) -> None:
        # Copy from: torch/testing/_internal/common_fsdp.py FSDPTest._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")

        backend = "gloo"

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        if torch.cuda.is_available() and torch.cuda.device_count():
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)

        self.reset_seed()

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        with (
            torch.backends.cudnn.flags(
                enabled=False, deterministic=True, benchmark=False
            ),
            patch.object(dist, "batch_isend_irecv", new=batch_isend_irecv_gloo),
            patch.object(dist, "all_to_all", new=all_to_all_gloo),
            patch.object(dist, "all_to_all_single", new=all_to_all_single_gloo),
            patch(
                "colossalai.pipeline.p2p._check_device",
                return_value=(torch.device("cuda"), False),
            ),
        ):
            torch.use_deterministic_algorithms(mode=True)
            self.run_test(test_name, pipe)

        try:
            dist.barrier()
        except RuntimeError:
            # Some processes may be hung due to synchronization error,
            # and return an error as other processes closed the connection.
            # Simply ignore the error here.
            pass

        dist.destroy_process_group()

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)


def batch_isend_irecv_gloo(p2p_op_list: list[dist.P2POp]) -> list[dist.Work]:
    reqs: list[tuple[dist.Work, torch.Tensor]] = []
    for p2p_op in p2p_op_list:
        if p2p_op.op == dist.isend:
            tensor = p2p_op.tensor.to("cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)
        else:
            tensor = torch.empty_like(p2p_op.tensor, device="cpu")
            work = p2p_op.op(tensor, p2p_op.peer, p2p_op.group, p2p_op.tag)

        reqs.append((work, tensor))

    send_reqs = []
    with torch.no_grad():
        for (req, tensor), p2p_op in zip(reqs, p2p_op_list):
            if req is None:
                continue

            if p2p_op.op == dist.irecv:
                req.wait()
                p2p_op.tensor.copy_(tensor)
            else:
                send_reqs.append(req)

    return send_reqs


def all_to_all_gloo(
    output_tensor_list: list[torch.Tensor],
    input_tensor_list: list[torch.Tensor],
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    """Backend gloo doesn't support all_to_all, so we simulate it here."""
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    # Each rank gathers the "i-th" input from all ranks
    for i in range(world_size):
        chunk = input_tensor_list[i].to("cpu")  # Each rank's i-th input tensor
        gathered_tensors = [
            torch.empty_like(chunk) for _ in range(world_size)
        ]  # Buffers for allgather

        # Perform all_gather to collect the i-th tensor from all ranks
        dist.all_gather(gathered_tensors, chunk, group)

        if i == rank:
            assert len(output_tensor_list) == len(gathered_tensors)
            for output_tensor, gathered_tensor in zip(
                output_tensor_list, gathered_tensors
            ):
                output_tensor.copy_(gathered_tensor)


def all_to_all_single_gloo(
    output: torch.Tensor,
    input: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: Optional[bool] = False,
):
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    chunk_size = input.size(0) // world_size
    gathered_tensors = [torch.empty_like(input) for _ in range(world_size)]

    dist.all_gather(gathered_tensors, input, group)

    output_tensor = torch.cat(
        [
            gathered_tensors[i][rank * chunk_size : (rank + 1) * chunk_size]
            for i in range(world_size)
        ],
        dim=0,
    )
    output.copy_(output_tensor)
