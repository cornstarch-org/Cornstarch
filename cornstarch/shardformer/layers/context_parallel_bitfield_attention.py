from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.kernel.bitfield_attention import (
    _flash_attn_backward,
    _flash_attn_forward,
)
from cornstarch.kernel.interface import repeat_kv


class ContextParallelBitfieldAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seqlen_qs: torch.Tensor,
        seqlen_ks: torch.Tensor,
        offsets: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        heads_stride: int = 1,
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
        assert (
            nheads % heads_stride == 0
        ), "number of heads must be divisible by heads_stride"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda

        # TODO: see the notion note. Need to gather k/v properly.

        gathered_k = [
            torch.empty(
                (batch, seqlen_k, heads_stride, d), dtype=k.dtype, device=k.device
            )
            for _ in range(sp_world_size)
        ]
        gathered_v = [
            torch.empty(
                (batch, seqlen_k, heads_stride, d), dtype=v.dtype, device=v.device
            )
            for _ in range(sp_world_size)
        ]

        gk_work = dist.all_gather(
            gathered_k,
            k[:, :, :heads_stride, :].contiguous(),
            group=sp_group,
            async_op=True,
        )
        gv_work = dist.all_gather(
            gathered_v,
            v[:, :, :heads_stride, :].contiguous(),
            group=sp_group,
            async_op=True,
        )

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        softmax_scales: list[torch.Tensor] = []

        head_index = 0
        while head_index < nheads:
            gk_work.wait()
            gv_work.wait()

            current_q = q[:, :, head_index : head_index + heads_stride, :].contiguous()
            current_k = torch.cat(gathered_k, dim=1).contiguous()
            current_v = torch.cat(gathered_v, dim=1).contiguous()

            head_index += heads_stride
            # prefetch next heads
            if head_index < nheads:
                gk_work = dist.all_gather(
                    gathered_k,
                    k[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                gv_work = dist.all_gather(
                    gathered_v,
                    v[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )

            o, lse, softmax_scale = _flash_attn_forward(
                current_q,
                current_k,
                current_v,
                bias=bias,
                softmax_scale=softmax_scale,
                mask=mask,
                seqlen_qs=seqlen_qs,
                seqlen_ks=seqlen_ks,
                offsets=offsets,
            )

            os.append(o)
            lses.append(lse)
            softmax_scales.append(softmax_scale)

        ctx.save_for_backward(q, k, v, bias, mask, seqlen_qs, seqlen_ks, offsets)
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scales = softmax_scales
        ctx.sp_group = sp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        q, k, v, bias, mask, seqlen_qs, seqlen_ks, offsets = ctx.saved_tensors
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scales: list[torch.Tensor] = ctx.softmax_scales
        sp_group: dist.ProcessGroup = ctx.sp_group
        sp_world_size = dist.get_world_size(sp_group)

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape

        gathered_k = [
            torch.empty(
                (batch, seqlen_k, heads_stride, d), dtype=k.dtype, device=k.device
            )
            for _ in range(sp_world_size)
        ]
        gathered_v = [
            torch.empty(
                (batch, seqlen_k, heads_stride, d), dtype=v.dtype, device=v.device
            )
            for _ in range(sp_world_size)
        ]

        gk_work = dist.all_gather(
            gathered_k,
            k[:, :, :heads_stride, :].contiguous(),
            group=sp_group,
            async_op=True,
        )
        gv_work = dist.all_gather(
            gathered_v,
            v[:, :, :heads_stride, :].contiguous(),
            group=sp_group,
            async_op=True,
        )

        dqs: list[torch.Tensor] = []
        dks: list[torch.Tensor] = []
        dvs: list[torch.Tensor] = []

        head_index = 0

        dq = torch.empty(
            (batch, seqlen_q, heads_stride, d), dtype=q.dtype, device=q.device
        )
        dk = torch.empty(
            (batch, seqlen_k, heads_stride, d), dtype=k.dtype, device=k.device
        )
        dv = torch.empty(
            (batch, seqlen_k, heads_stride, d), dtype=v.dtype, device=v.device
        )
        dgk = torch.empty(
            (batch, seqlen_k * sp_world_size, heads_stride, d),
            dtype=k.dtype,
            device=k.device,
        )
        dgv = torch.empty(
            (batch, seqlen_k * sp_world_size, heads_stride, d),
            dtype=v.dtype,
            device=v.device,
        )

        while head_index < nheads:
            gk_work.wait()
            gv_work.wait()

            current_q = q[:, :, head_index : head_index + heads_stride, :]
            current_k = torch.cat(gathered_k, dim=1)
            current_v = torch.cat(gathered_v, dim=1)

            head_index += heads_stride
            # prefetch next heads
            if head_index < nheads:
                gk_work = dist.all_gather(
                    gathered_k,
                    k[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                gv_work = dist.all_gather(
                    gathered_v,
                    v[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )

            o = os.pop(0)
            lse = lses.pop(0)
            softmax_scale = softmax_scales.pop(0)

            _flash_attn_backward(
                do[:, :, head_index - heads_stride : head_index, :],
                current_q,
                current_k,
                current_v,
                o,
                lse,
                dq,
                dgk,
                dgv,
                bias=bias,
                softmax_scale=softmax_scale,
                mask=mask,
                seqlen_qs=seqlen_qs,
                seqlen_ks=seqlen_ks,
                offsets=offsets,
            )

            dist.reduce_scatter(
                dk, list(dgk.chunk(sp_world_size, dim=1)), group=sp_group
            )
            dist.reduce_scatter(
                dv, list(dgv.chunk(sp_world_size, dim=1)), group=sp_group
            )

            dqs.append(dq.clone())
            dks.append(dk.clone())
            dvs.append(dv.clone())

        return (
            torch.cat(dqs, dim=2),
            torch.cat(dks, dim=2),
            torch.cat(dvs, dim=2),
            *[None] * 8,
        )


def context_parallel_bitfield_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    sp_group: dist.ProcessGroup,
    seqlen_qs: torch.Tensor,
    seqlen_ks: torch.Tensor,
    offsets: torch.Tensor,
    heads_stride: int = 1,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert (
        attention_mask is not None and attention_mask.dtype == torch.int64
    ), "Bitfield attention requires an attention mask of type torch.int64."

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # BAM follows FA2 that uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelBitfieldAttention.apply(
        query,
        key,
        value,
        attention_mask,
        sp_group,
        seqlen_qs,
        seqlen_ks,
        offsets,
        None,
        None,
        heads_stride,
    )

    return attn_output, None
