from typing import Optional

import torch
import torch.distributed as dist

from cornstarch.kernel.bitfield_attention import (
    _flash_attn_backward,
    _flash_attn_forward,
)


class ContextParallelBitfieldAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        bias: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
        heads_k_stride: number of key/value heads to transfer
        in a single pipelined all-gather operation.
        """
        # shape constraints
        sp_world_size = dist.get_world_size(sp_group)
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert mask.shape == (
            batch,
            seqlen_k * sp_world_size,
        ), f"Expected mask shape ({batch}, {seqlen_k * sp_world_size}), but got {mask.shape}"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda

        gathered_k = [torch.empty_like(k) for _ in range(sp_world_size)]
        gathered_v = [torch.empty_like(v) for _ in range(sp_world_size)]

        dist.all_gather(gathered_k, k, group=sp_group)
        dist.all_gather(gathered_v, v, group=sp_group)

        gathered_k = torch.cat(gathered_k, dim=1).contiguous()
        gathered_v = torch.cat(gathered_v, dim=1).contiguous()

        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, gathered_k, gathered_v, bias=bias, softmax_scale=softmax_scale, mask=mask
        )

        ctx.save_for_backward(q, k, v, o, lse, bias, mask)
        ctx.sp_group = sp_group

        return o

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        q, k, v, o, lse, bias, mask = ctx.saved_tensors
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        sp_group: dist.ProcessGroup = ctx.sp_group
        sp_world_size = dist.get_world_size(sp_group)

        gathered_k = [torch.empty_like(k) for _ in range(sp_world_size)]
        gathered_v = [torch.empty_like(v) for _ in range(sp_world_size)]

        dist.all_gather(gathered_k, k, group=sp_group)
        dist.all_gather(gathered_v, v, group=sp_group)

        gathered_k = torch.cat(gathered_k, dim=1)
        gathered_v = torch.cat(gathered_v, dim=1)

        dq = torch.empty_like(q)
        dgk = torch.empty_like(gathered_k)
        dgv = torch.empty_like(gathered_v)
        _flash_attn_backward(
            do,
            q,
            gathered_k,
            gathered_v,
            o,
            lse,
            dq,
            dgk,
            dgv,
            bias=bias,
            softmax_scale=ctx.softmax_scale,
            mask=mask,
        )

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        dist.reduce_scatter(
            dk, dgk.chunk(sp_world_size, dim=1), op=dist.ReduceOp.SUM, group=sp_group
        )
        dist.reduce_scatter(
            dv, dgv.chunk(sp_world_size, dim=1), op=dist.ReduceOp.SUM, group=sp_group
        )

        return dq, dk, dv, None, None, None, None


context_parallel_bitfield_attn_func = ContextParallelBitfieldAttention.apply
