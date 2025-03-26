from __future__ import annotations

import heapq
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

from cornstarch.kernel.attention import flash_attn_func
from cornstarch.kernel.bitfield_attention import bitfield_attn_func
from cornstarch.kernel.interface import BitfieldUtils
from cornstarch.shardformer.layers.context_parallel_attention import (
    ContextParallelCornstarchAttention,
)
from cornstarch.shardformer.layers.context_parallel_bitfield_attention import (
    ContextParallelBitfieldAttention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

from ..distributed_base import GlooDistributedTestBase


def get_causal_assignments(num_blocks: int):
    """
    Assignment pattern for causal mask looks like:
    [..., 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]
    """
    base = [0, 1, 2, 3]
    tail = [3, 2, 1, 0]

    assignments = tail[:] if num_blocks >= 4 else tail[:num_blocks]
    current_length = len(assignments)
    toggle = True
    while current_length < num_blocks:
        needed = num_blocks - current_length
        prepend = base[:] if toggle else tail[:]
        if needed < 4:
            prepend = prepend[-needed:]
        assignments = prepend + assignments
        current_length = len(assignments)
        toggle = not toggle

    assignments = torch.as_tensor(assignments, device="cuda").repeat_interleave(128)

    return assignments


def get_full_assignments(num_blocks: int):
    pattern = [0, 1, 2, 3]
    assignments = (
        pattern * (num_blocks // len(pattern)) + pattern[: num_blocks % len(pattern)]
    )

    assignments = torch.as_tensor(assignments, device="cuda").repeat_interleave(128)

    return assignments


class TestContextParallelBatchSplitUtilClass:

    num_heads: int
    head_dim: int

    @classmethod
    def setup_class(cls: TestContextParallelBatchSplitUtilClass):
        cls.num_heads = 8
        cls.head_dim = 64

    @pytest.fixture(autouse=True)
    def cleanup(self):
        self.teardown_method()

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

        ContextParallelBatchSplitUtils.clear_cache()

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_uniform_singlebatch(self, seqlen: int, world_size: int):
        batch_size = 1
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            ContextParallelBatchSplitUtils.create_context_parallel_split_uniform(
                mask,
                sp_group=dist.GroupMember.WORLD,
            )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen < 128 * world_size:
                start_offset, end_offset = 0, seqlen
            else:
                expected_seqlen_per_rank = math.ceil(seqlen / world_size)
                start_offset = expected_seqlen_per_rank * rank
                end_offset = min(expected_seqlen_per_rank * (rank + 1), seqlen)

            torch.testing.assert_close(data[:, start_offset:end_offset], split_data)

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_zigzag_singlebatch(self, seqlen: int, world_size: int):
        batch_size: int = 1
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            ContextParallelBatchSplitUtils.create_context_parallel_split_zigzag(
                mask,
                sp_group=dist.GroupMember.WORLD,
            )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen < 128 * world_size:
                expected_split_data = data
                expected_indices = np.arange(seqlen)
            else:
                num_chunks = world_size * 2
                base_size = seqlen // num_chunks
                remainder = seqlen % num_chunks
                chunk_sizes = [
                    base_size + 1 if i < remainder else base_size
                    for i in range(num_chunks)
                ]

                indices = np.arange(seqlen)
                chunks = []
                start = 0
                for size in chunk_sizes:
                    chunks.append(indices[start : start + size])
                    start += size

                # Each rank should get its corresponding chunk and the symmetric one.
                expected_first = chunks[rank]
                expected_second = chunks[-rank - 1]
                expected_indices = np.concatenate([expected_first, expected_second])

                expected_split_data = data[:, expected_indices, :, :]

            torch.testing.assert_close(expected_split_data, split_data)

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    @pytest.mark.parametrize("mask_type", ["causal", "full"])
    def test_split_batch_makespan_min_singlebatch(
        self, seqlen: int, world_size: int, mask_type: str
    ):
        batch_size = 1

        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )

        if mask_type == "causal":
            mask = torch.tril(
                torch.ones(
                    (batch_size, seqlen, seqlen), dtype=torch.bool, device="cuda"
                )
            )
        elif mask_type == "full":
            mask = torch.ones(
                (batch_size, seqlen, seqlen), dtype=torch.bool, device="cuda"
            )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            ContextParallelBatchSplitUtils.create_context_parallel_split_makespan_minimization(
                mask,
                sp_group=dist.GroupMember.WORLD,
            )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen <= 128 * 4:
                expected_split_data = data
                expected_indices = torch.arange(seqlen, device="cuda")
            else:
                num_blocks = math.ceil(seqlen / 128)

                if mask_type == "causal":
                    assignments = get_causal_assignments(num_blocks)
                elif mask_type == "full":
                    assignments = get_full_assignments(num_blocks)

                expected_indices = torch.nonzero(assignments == rank, as_tuple=True)[0]
                expected_indices = expected_indices[expected_indices < data.shape[1]]

                expected_split_data = data[:, expected_indices]

            torch.testing.assert_close(expected_split_data, split_data)


@instantiate_parametrized_tests
class TestBitfieldContextParallelismClass(GlooDistributedTestBase):

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

        ref_out: torch.Tensor = bitfield_attn_func(
            query, key, value, bitfield_mask=mask
        )

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

        offsets_per_rank = torch.chunk(
            torch.arange(query.shape[1], device="cuda"), self.world_size, dim=0
        )

        cp_out: torch.Tensor = ContextParallelBitfieldAttention.apply(
            local_query,
            local_key,
            local_value,
            mask,
            offsets_per_rank,
            dist.GroupMember.WORLD,
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

        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
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


@instantiate_parametrized_tests
class TestFlashAttentionWithMaskContextParallelismClass(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2, 4], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [64, 256, 336, 400, 1024], name_fn=lambda x: f"seq={x}")
    def test(self, batch_size: int, seq_len: int):
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.bfloat16
            ).normal_(mean=0, std=0.5),
        )

        for t in [query, key, value]:
            t.requires_grad_()

        mask = torch.tril(
            torch.full(
                (batch_size, seq_len, seq_len), True, dtype=torch.bool, device="cuda"
            )
        )

        ref_out: torch.Tensor = flash_attn_func(query, key, value, mask=mask)

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

        local_mask = torch.chunk(mask, self.world_size, dim=1)[self.rank]

        assert (
            local_key.shape
            == local_value.shape
            == (batch_size, seq_len // self.world_size, 8, 64)
        )

        seqlen_per_rank = torch.tensor(
            [seq_len // self.world_size] * self.world_size, device="cuda"
        )

        cp_out: torch.Tensor = ContextParallelCornstarchAttention.apply(
            local_query,
            local_key,
            local_value,
            local_mask,
            seqlen_per_rank,
            dist.GroupMember.WORLD,
        )

        torch.testing.assert_close(
            torch.chunk(ref_out, self.world_size, dim=1)[self.rank],
            cp_out,
            rtol=5e-3,
            atol=5e-3,
        )

        # ========================================================================
        # Check backward
        # ========================================================================

        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
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
