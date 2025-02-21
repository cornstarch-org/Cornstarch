import math

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
    @pytest.mark.parametrize("batch_size", [1, 2], ids=["bs=1", "bs=2"])
    @pytest.mark.parametrize(
        "seqlen",
        [57, 64, 256, 335, 402, 1024],
        ids=["seq=57", "seq=64", "seq=256", "seq=335", "seq=402", "seq=1024"],
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_uniform(self, batch_size: int, seqlen: int, world_size: int):
        num_heads = 8
        head_dim = 64

        data = torch.randn((batch_size, seqlen, num_heads, head_dim), device="cuda")
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
