from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.kernel.bitfield_attention import bitfield_attn_func
from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_bitfield_attn_func,
)
from cornstarch.shardformer.layers.utils import ContextParallelBatchSplitUtils

from ..distributed_base import GlooDistributedTestBase


class TestContextParallelBatchSplitUtilClass:

    num_heads: int
    head_dim: int

    @classmethod
    def setup_class(cls: TestContextParallelBatchSplitUtilClass):
        cls.num_heads = 8
        cls.head_dim = 64

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

        ContextParallelBatchSplitUtils.clear_split_cache()

    @pytest.mark.parametrize("batch_size", [1, 2], ids=["bs=1", "bs=2"])
    @pytest.mark.parametrize(
        "seqlen",
        [57, 64, 256, 335, 402, 1024],
        ids=["seq=57", "seq=64", "seq=256", "seq=335", "seq=402", "seq=1024"],
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_uniform(self, batch_size: int, seqlen: int, world_size: int):
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        # Expected data structure
        expected_seqlen_per_rank = math.ceil(seqlen / world_size)

        store = FakeStore()
        for rank in range(world_size):
            if dist.is_initialized():
                dist.destroy_process_group()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            split_data = ContextParallelBatchSplitUtils._split_batch_uniform(
                data, mask, sp_group=dist.GroupMember.WORLD
            )

            start_offset = expected_seqlen_per_rank * rank
            end_offset = min(expected_seqlen_per_rank * (rank + 1), seqlen)
            torch.testing.assert_close(
                data[:, start_offset:end_offset, :, :], split_data
            )

    @pytest.mark.parametrize("batch_size", [1, 2], ids=["bs=1", "bs=2"])
    @pytest.mark.parametrize(
        "seqlen",
        [57, 64, 256, 335, 402, 1024],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_zigzag(self, batch_size: int, seqlen: int, world_size: int):
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        # Expected data structure for world_size=4
        num_chunks = world_size * 2
        base_size = seqlen // num_chunks
        remainder = seqlen % num_chunks
        chunk_sizes = [
            base_size + 1 if i < remainder else base_size for i in range(num_chunks)
        ]

        indices = np.arange(seqlen)
        chunks = []
        start = 0
        for size in chunk_sizes:
            chunks.append(indices[start : start + size])
            start += size

        store = FakeStore()
        for rank in range(world_size):
            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            split_data = ContextParallelBatchSplitUtils._split_batch_zigzag(
                data, mask, sp_group=dist.GroupMember.WORLD
            )

            if world_size == 1:
                expected_split_data = data
            else:
                # Each rank should get its corresponding chunk and the symmetric one.
                expected_first = (
                    chunks[rank]
                    if rank < len(chunks)
                    else np.array([], dtype=indices.dtype)
                )
                expected_second = (
                    chunks[-rank - 1]
                    if rank < len(chunks)
                    else np.array([], dtype=indices.dtype)
                )
                expected_indices = np.concatenate([expected_first, expected_second])

                expected_split_data = data[:, expected_indices, :, :]

            torch.testing.assert_close(expected_split_data, split_data)

            self.teardown_method()

    @pytest.mark.parametrize("batch_size", [1, 2], ids=["bs=1", "bs=2"])
    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    @pytest.mark.parametrize("mask_type", ["causal", "full"])
    def test_split_batch_makespan_min(
        self, batch_size: int, seqlen: int, world_size: int, mask_type: str
    ):
        torch.manual_seed(0)

        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )

        if mask_type == "causal":
            mask = torch.full(
                (batch_size, seqlen), (1 << 62) | 1, dtype=torch.int64, device="cuda"
            )
        else:
            mask = torch.full(
                (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
            )

        # Expected data structure for world_size=4

        store = FakeStore()
        for rank in range(world_size):
            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            split_data = (
                ContextParallelBatchSplitUtils._split_batch_makespan_minimization(
                    data, mask, sp_group=dist.GroupMember.WORLD
                )
            )

            if world_size == 1 or seqlen <= 128 * 4:
                expected_split_data = data
            else:
                num_blocks = math.ceil(seqlen / 128)

                if mask_type == "causal":
                    raise NotImplementedError
                else:
                    pattern = [0, 1, 2, 3]
                    assignments = (
                        pattern * (num_blocks // len(pattern))
                        + pattern[: num_blocks % len(pattern)]
                    )
                    assignments = torch.as_tensor(
                        assignments, device="cuda"
                    ).repeat_interleave(128)

                indices = torch.nonzero(assignments == rank, as_tuple=True)[0]
                expected_split_data = torch.index_select(data, dim=1, index=indices)

            torch.testing.assert_close(expected_split_data, split_data)

            self.teardown_method()


@instantiate_parametrized_tests
class TestContextParallelismClass(GlooDistributedTestBase):

    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2, 4], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [64, 256, 336, 400, 1024], name_fn=lambda x: f"seq={x}")
    def test(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, ...]:
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.bfloat16
            ).normal_(mean=0, std=0.5),
        )

        for t in [query, key, value]:
            t.requires_grad_()

        # mask = torch.full(
        #     (batch_size, seq_len), 1 << 1, dtype=torch.int64, device="cuda"
        # )
        mask = torch.full(
            (batch_size, seq_len),
            (1 << 62) | 1 | (1 << 1),
            dtype=torch.int64,
            device="cuda",
        )
        mask[:, 32:44] = 1 << 1
        if seq_len >= 256:
            mask[:, 180:280] = 1 << 1

        ref_out: torch.Tensor = bitfield_attn_func(query, key, value, None, None, mask)

        local_query = (
            torch.chunk(query, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_key = (
            torch.chunk(key, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_value = (
            torch.chunk(value, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        assert (
            local_key.shape
            == local_value.shape
            == (batch_size, seq_len // self.world_size, 8, 64)
        )

        cp_out: torch.Tensor = context_parallel_bitfield_attn_func(
            local_query,
            local_key,
            local_value,
            dist.GroupMember.WORLD,
            None,
            None,
            mask,
        )

        torch.testing.assert_close(
            torch.chunk(ref_out, self.world_size, dim=1)[self.rank].contiguous(),
            cp_out,
            rtol=5e-3,
            atol=5e-3,
        )

        # ========================================================================
        # Check backward
        # ========================================================================

        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.05)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, [query, key, value], dout)

        cp_dout = torch.chunk(dout, self.world_size, dim=1)[self.rank].contiguous()
        cp_dq, cp_dk, cp_dv = torch.autograd.grad(
            cp_out, [local_query, local_key, local_value], cp_dout
        )

        torch.testing.assert_close(
            torch.chunk(ref_dq, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dq,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dk, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dk,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dv, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dv,
            rtol=5e-3,
            atol=5e-3,
        )
