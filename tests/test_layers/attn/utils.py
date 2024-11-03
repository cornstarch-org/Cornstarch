import os
import random
import re
import sys
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from gloo_utils import batch_isend_irecv_gloo
from torch.testing import assert_close
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA


class RingAttentionTestBase(MultiProcessTestCase):

    batch_size = 4
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    def prepare_qkv_for_flash_attention(
        self,
        kernel_impl: str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare q, k, v tensors for flash attention.
        Output from flash attention is used as a base for ring attention tests.

        Returns:
            torch.Tensor: query
            torch.Tensor: key
            torch.Tensor: value
            torch.Tensor: dout (fake gradient for backward pass)
        """
        qkv = torch.randn(
            3,
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            device="cuda",
            dtype=dtype,
        ).normal_(mean=0.0, std=0.5)
        dout = torch.randn(
            self.batch_size,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            device="cuda",
            dtype=dtype,
        )

        if kernel_impl == "triton":
            # triton requires q, k, v and dout to be in shape of [batch_size, num_heads, seq_len, head_dim]
            qkv = qkv.transpose(2, 3)
            dout = dout.transpose(1, 2)

        q, k, v = qkv.chunk(3, dim=0)
        q = q.squeeze(dim=0).detach().clone().contiguous()
        k = k.squeeze(dim=0).detach().clone().contiguous()
        v = v.squeeze(dim=0).detach().clone().contiguous()
        for tensor in [q, k, v, dout]:
            tensor.requires_grad_(True)
            tensor.retain_grad()

        return q, k, v, dout

    def prepare_qkv_for_ring_attention(
        self,
        qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dout: torch.Tensor,
        seq_dim: int,
    ) -> tuple[torch.Tensor, ...]:
        """
        Prepare rank-local q, k, v tensors for ring attention.
        Output will be compared with flash attention output.

        Returns:
            torch.Tensor: query
            torch.Tensor: key
            torch.Tensor: value
            torch.Tensor: dout (fake gradient for backward pass)
        """
        q, k, v = qkv
        local_q = (
            q.chunk(self.world_size, dim=seq_dim)[self.rank]
            .detach()
            .clone()
            .contiguous()
        )
        local_k = (
            k.chunk(self.world_size, dim=seq_dim)[self.rank]
            .detach()
            .clone()
            .contiguous()
        )
        local_v = (
            v.chunk(self.world_size, dim=seq_dim)[self.rank]
            .detach()
            .clone()
            .contiguous()
        )
        local_dout = (
            dout.chunk(self.world_size, dim=seq_dim)[self.rank]
            .detach()
            .clone()
            .contiguous()
        )

        for tensor in [local_q, local_k, local_v, local_dout]:
            tensor.requires_grad_(True)
            tensor.retain_grad()

        return local_q, local_k, local_v, local_dout

    def check_tensors(
        self,
        reference_tensors: tuple[torch.Tensor, ...],
        test_tensors: tuple[torch.Tensor, ...],
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        assert len(reference_tensors) == len(test_tensors)
        for ref, test in zip(reference_tensors, test_tensors):
            assert_close(ref, test, rtol=rtol, atol=atol)

    _world_size: int

    @property
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def setUp(self) -> None:
        super().setUp()

        # Find if parameterized world_size exists in the test method name
        match = re.search(r"world_size_(\d+)", self._testMethodName)
        world_size = match.group(1) if match else "8"

        with patch.dict(
            os.environ,
            {
                "CUBLAS_WORKSPACE_CONFIG": ":16:8",
                "WORLD_SIZE": world_size,
            },
        ):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    @classmethod
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs
    ) -> None:
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
                enabled=True, deterministic=True, benchmark=False
            ),
            patch.object(dist, "batch_isend_irecv", new=batch_isend_irecv_gloo),
        ):
            self.run_test(test_name, parent_pipe)

        dist.barrier()

        dist.destroy_process_group()

    def reset_seed(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
