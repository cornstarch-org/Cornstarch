import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.kernel.bitfield_attention import bitfield_attn_func
from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_bitfield_attn_func,
)

from ..distributed_base import GlooDistributedTestBase


def test_fake_context_parallelism():
    batch_size: int = 1
    seq_len: int = 128

    query, key, value = torch.unbind(
        torch.randn(
            (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.float16
        ).normal_(mean=0, std=0.5),
    )

    for t in [query, key, value]:
        t.requires_grad_()

    mask = torch.full((batch_size, seq_len), 1 << 1, dtype=torch.int64, device="cuda")

    ref_out: torch.Tensor = bitfield_attn_func(query, key, value, None, None, mask)

    local_queries = [q.requires_grad_().contiguous() for q in query.chunk(2, dim=1)]

    cp_outs: list[torch.Tensor] = [
        bitfield_attn_func(local_query, key, value, None, None, mask)
        for local_query in local_queries
    ]

    # torch.testing.assert_close(ref_out.chunk(2, dim=1)[0], cp_out, rtol=5e-3, atol=5e-3)

    # ========================================================================
    # Check backward
    # ========================================================================

    dout = torch.randn_like(ref_out)
    cp_douts = [do.contiguous() for do in dout.chunk(2, dim=1)]

    ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, [query, key, value], dout)

    assert key.grad is None and value.grad is None

    cp_dq0, cp_dk0, cp_dv0 = torch.autograd.grad(
        cp_outs[0], [local_queries[0], key, value], cp_douts[0]
    )

    cp_dq1, cp_dk1, cp_dv1 = torch.autograd.grad(
        cp_outs[1], [local_queries[1], key, value], cp_douts[1]
    )

    print("done")

    # torch.testing.assert_close(ref_dq.chunk(2, dim=1)[0], cp_dq, rtol=5e-3, atol=5e-3)
    # torch.testing.assert_close(ref_dk, cp_dk, rtol=5e-3, atol=5e-3)
    # torch.testing.assert_close(ref_dv, cp_dv, rtol=5e-3, atol=5e-3)


@instantiate_parametrized_tests
class TestContextParallelismClass(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2, 4], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [64, 256, 400, 1024], name_fn=lambda x: f"seq={x}")
    def test(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, ...]:
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.bfloat16
            ).normal_(mean=0, std=0.5),
        )

        for t in [query, key, value]:
            t.requires_grad_()

        mask = torch.full(
            (batch_size, seq_len), 1 << 1, dtype=torch.int64, device="cuda"
        )

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

        dout = torch.randn_like(ref_out)
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
