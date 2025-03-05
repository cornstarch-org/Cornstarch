from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.kernel.bitfield_attention import (
    _bitfield_attn_backward,
    _bitfield_attn_forward,
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
        seqlen_qs: list[torch.Tensor],
        seqlen_ks: torch.Tensor,
        offsets: list[torch.Tensor],
        indices_perm: torch.Tensor,
        indices_inverse_perm: torch.Tensor,
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
        sp_rank = dist.get_rank(sp_group)

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape
        assert k.shape == (batch, seqlen_k, nheads, d)
        assert v.shape == (batch, seqlen_k, nheads, d)
        assert (
            nheads % heads_stride == 0
        ), "number of heads must be divisible by heads_stride"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda

        local_seqlen_per_rank = [
            max(seqlen_qs_per_rank).item() for seqlen_qs_per_rank in seqlen_qs
        ]
        assert mask.shape == (
            batch,
            sum(local_seqlen_per_rank),
        ), f"Expected mask shape ({batch}, {sum(local_seqlen_per_rank)}), but got {mask.shape}"

        gathered_k = [
            torch.empty(
                (batch, local_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )
            for local_seqlen in local_seqlen_per_rank
        ]
        gathered_v = [
            torch.empty(
                (batch, local_seqlen, heads_stride, d),
                dtype=v.dtype,
                device=v.device,
            )
            for local_seqlen in local_seqlen_per_rank
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

        unsqueezed_indices_perm = (
            indices_perm.unsqueeze(2).unsqueeze(3).expand(-1, -1, heads_stride, d)
        )

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        softmax_scales: list[torch.Tensor] = []

        head_index = 0
        while head_index < nheads:
            gk_work.wait()
            gv_work.wait()

            current_q = q[:, :, head_index : head_index + heads_stride, :].contiguous()
            current_k = torch.cat(gathered_k, dim=1)
            current_k = torch.gather(
                current_k, dim=1, index=unsqueezed_indices_perm
            ).contiguous()
            current_v = torch.cat(gathered_v, dim=1)
            current_v = torch.gather(
                current_v, dim=1, index=unsqueezed_indices_perm
            ).contiguous()

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

            o, lse, softmax_scale = _bitfield_attn_forward(
                current_q,
                current_k,
                current_v,
                bias=bias,
                softmax_scale=softmax_scale,
                mask=mask,
                seqlen_qs=seqlen_qs[sp_rank],
                seqlen_ks=seqlen_ks,
                offsets=offsets[sp_rank],
            )

            os.append(o)
            lses.append(lse)
            softmax_scales.append(softmax_scale)

        ctx.save_for_backward(
            q,
            k,
            v,
            bias,
            mask,
            seqlen_qs[sp_rank],
            seqlen_ks,
            offsets[sp_rank],
            indices_perm,
            indices_inverse_perm,
        )
        ctx.local_seqlen_per_rank = local_seqlen_per_rank
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scales = softmax_scales
        ctx.sp_group = sp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        (
            q,
            k,
            v,
            bias,
            mask,
            seqlen_qs,
            seqlen_ks,
            offsets,
            indices_perm,
            indices_inverse_perm,
        ) = ctx.saved_tensors
        local_seqlen_per_rank = ctx.local_seqlen_per_rank
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scales: list[torch.Tensor] = ctx.softmax_scales
        sp_group: dist.ProcessGroup = ctx.sp_group

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_k, _, _ = k.shape

        gathered_k = [
            torch.empty(
                (batch, local_seqlen, heads_stride, d), dtype=k.dtype, device=k.device
            )
            for local_seqlen in local_seqlen_per_rank
        ]
        gathered_v = [
            torch.empty(
                (batch, local_seqlen, heads_stride, d), dtype=v.dtype, device=v.device
            )
            for local_seqlen in local_seqlen_per_rank
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

        unsqueezed_indices_perm = (
            indices_perm.unsqueeze(2).unsqueeze(3).expand(-1, -1, heads_stride, d)
        )
        unsqueezed_indices_inverse_perm = (
            indices_inverse_perm.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, heads_stride, d)
        )

        dq = torch.empty(
            (batch, seqlen_q, heads_stride, d), dtype=q.dtype, device=q.device
        )
        dk = torch.empty(
            (batch, seqlen_k, heads_stride, d), dtype=k.dtype, device=k.device
        )
        dv = torch.empty(
            (batch, seqlen_k, heads_stride, d), dtype=v.dtype, device=v.device
        )
        dgk = torch.full(
            (batch, sum(local_seqlen_per_rank), heads_stride, d),
            torch.nan,
            dtype=k.dtype,
            device=k.device,
        )
        dgv = torch.full(
            (batch, sum(local_seqlen_per_rank), heads_stride, d),
            torch.nan,
            dtype=v.dtype,
            device=v.device,
        )

        while head_index < nheads:
            gk_work.wait()
            gv_work.wait()

            current_q = q[:, :, head_index : head_index + heads_stride, :]
            current_k = torch.cat(gathered_k, dim=1)
            current_k = torch.gather(
                current_k, dim=1, index=unsqueezed_indices_perm
            ).contiguous()
            current_v = torch.cat(gathered_v, dim=1)
            current_v = torch.gather(
                current_v, dim=1, index=unsqueezed_indices_perm
            ).contiguous()

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

            _bitfield_attn_backward(
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

            dgk = torch.gather(dgk, dim=1, index=unsqueezed_indices_inverse_perm)
            dgv = torch.gather(dgv, dim=1, index=unsqueezed_indices_inverse_perm)

            dist.reduce_scatter(
                dk, list(dgk.split(local_seqlen_per_rank, dim=1)), group=sp_group
            )
            dist.reduce_scatter(
                dv, list(dgv.split(local_seqlen_per_rank, dim=1)), group=sp_group
            )

            dqs.append(dq.clone())
            dks.append(dk.clone())
            dvs.append(dv.clone())

        return (
            torch.cat(dqs, dim=2),
            torch.cat(dks, dim=2),
            torch.cat(dvs, dim=2),
            *[None] * 10,
        )


def context_parallel_bitfield_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    sp_group: dist.ProcessGroup,
    seqlen_qs: list[torch.Tensor],
    seqlen_ks: torch.Tensor,
    offsets: list[torch.Tensor],
    indices_perm: torch.Tensor,
    indices_inverse_perm: torch.Tensor,
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
        indices_perm,
        indices_inverse_perm,
        None,
        None,
        heads_stride,
    )

    return attn_output, None
