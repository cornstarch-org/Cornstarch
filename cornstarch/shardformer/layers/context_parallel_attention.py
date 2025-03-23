from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.kernel.attention import _flash_attn_backward, _flash_attn_forward
from cornstarch.kernel.interface import repeat_kv


class ContextParallelAttentionWithMask(torch.autograd.Function):

    stream: torch.cuda.Stream = None

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        seqlen_per_rank: torch.Tensor,
        sp_group: dist.ProcessGroup,
        bias: Optional[torch.Tensor] = None,
        heads_stride: int = 1,
    ) -> torch.Tensor:
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_kv, nheads, headdim)
        mask: (batch_size, seqlen_q, total_seqlen)
            columns should be aligned with the order of gathered kv
        seqlen_per_rank: (world_size(sp_group),)
        """
        if ContextParallelAttentionWithMask.stream is None:
            ContextParallelAttentionWithMask.stream = torch.cuda.Stream()

        stream = ContextParallelAttentionWithMask.stream

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape

        assert mask.shape == (batch, seqlen_q, seqlen_per_rank.sum().item())
        assert k.shape == (batch, seqlen_kv, nheads, d)
        assert v.shape == (batch, seqlen_kv, nheads, d)
        assert (
            nheads % heads_stride == 0
        ), "number of heads must be divisible by heads_stride"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda

        per_head_events: list[torch.cuda.Event] = []

        # pre-allocate memory for k and v gathering for all heads
        total_seqlen = seqlen_per_rank.sum().item()
        gathered_kv = [
            torch.empty(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )
            for _ in range(nheads // heads_stride)
        ]

        # Initialize allgather works asynchronously
        with torch.cuda.stream(stream):
            for head_index in range(0, nheads, heads_stride):
                dist.all_gather(
                    gathered_kv[head_index // heads_stride][0].split(
                        seqlen_per_rank.tolist(), dim=1
                    ),
                    k[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                dist.all_gather(
                    gathered_kv[head_index // heads_stride][1].split(
                        seqlen_per_rank.tolist(), dim=1
                    ),
                    v[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                event = torch.cuda.Event()
                event.record(stream)
                per_head_events.append(event)

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        softmax_scales: list[torch.Tensor] = []

        assert len(per_head_events) == nheads // heads_stride
        for head_index in range(0, nheads, heads_stride):
            event = per_head_events[head_index // heads_stride]
            torch.cuda.current_stream().wait_event(event)

            current_q = q[:, :, head_index : head_index + heads_stride, :].contiguous()
            current_k = gathered_kv[head_index // heads_stride][0]
            current_v = gathered_kv[head_index // heads_stride][1]
            assert current_k.is_contiguous() and current_v.is_contiguous()

            o, lse, softmax_scale = _flash_attn_forward(
                current_q, current_k, current_v, bias, mask
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
            seqlen_per_rank,
        )
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scales = softmax_scales
        ctx.sp_group = sp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        stream = ContextParallelAttentionWithMask.stream

        q, k, v, bias, mask, seqlen_per_rank = ctx.saved_tensors
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scales: list[torch.Tensor] = ctx.softmax_scales
        sp_group: dist.ProcessGroup = ctx.sp_group

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape

        per_head_events: list[torch.cuda.Event] = []

        # pre-allocate memory for k and v gathering for all heads
        total_seqlen = seqlen_per_rank.sum().item()
        gathered_kv = [
            torch.empty(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )
            for _ in range(nheads // heads_stride)
        ]

        # Initialize allgather works asynchronously
        with torch.cuda.stream(stream):
            for head_index in range(0, nheads, heads_stride):
                dist.all_gather(
                    gathered_kv[head_index // heads_stride][0].split(
                        seqlen_per_rank.tolist(), dim=1
                    ),
                    k[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                dist.all_gather(
                    gathered_kv[head_index // heads_stride][1].split(
                        seqlen_per_rank.tolist(), dim=1
                    ),
                    v[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                event = torch.cuda.Event()
                event.record(stream)
                per_head_events.append(event)

        dqs: list[torch.Tensor] = []
        dks: list[torch.Tensor] = []
        dvs: list[torch.Tensor] = []

        assert len(per_head_events) == nheads // heads_stride
        for head_index in range(0, nheads, heads_stride):
            event = per_head_events[head_index // heads_stride]
            torch.cuda.current_stream().wait_event(event)

            dq = torch.empty(
                (batch, seqlen_q, heads_stride, d),
                dtype=q.dtype,
                device=q.device,
            )
            dkv = torch.empty(
                (2, batch, seqlen_kv, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )

            dgkv = torch.full(
                (2, batch, total_seqlen, heads_stride, d),
                torch.nan,
                dtype=k.dtype,
                device=k.device,
            )

            _flash_attn_backward(
                do=do[:, :, head_index : head_index + heads_stride, :],
                q=q[:, :, head_index : head_index + heads_stride, :],
                k=gathered_kv[head_index // heads_stride][0],
                v=gathered_kv[head_index // heads_stride][1],
                o=os[head_index // heads_stride],
                lse=lses[head_index // heads_stride],
                dq=dq,
                dk=dgkv[0],
                dv=dgkv[1],
                bias=bias,
                mask=mask,
                softmax_scale=softmax_scales[head_index // heads_stride],
            )

            dist.reduce_scatter(
                dkv[0], dgkv[0].split(seqlen_per_rank.tolist(), dim=1), group=sp_group
            )
            dist.reduce_scatter(
                dkv[1], dgkv[1].split(seqlen_per_rank.tolist(), dim=1), group=sp_group
            )

            dqs.append(dq.clone())
            dks.append(dkv[0].clone())
            dvs.append(dkv[1].clone())

        return (
            torch.cat(dqs, dim=2),
            torch.cat(dks, dim=2),
            torch.cat(dvs, dim=2),
            None,
            None,
            None,
            None,
            None,
        )


def context_parallel_flash_attention_with_mask_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    seqlen_per_rank: torch.Tensor,
    sp_group: dist.ProcessGroup,
    bias: Optional[torch.Tensor] = None,
    heads_stride: int = 1,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is not None and attention_mask.dtype == torch.bool

    if query.shape[1] > key.shape[1]:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # FA1 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelAttentionWithMask.apply(
        query, key, value, attention_mask, seqlen_per_rank, sp_group, bias, heads_stride
    )

    return attn_output, None
