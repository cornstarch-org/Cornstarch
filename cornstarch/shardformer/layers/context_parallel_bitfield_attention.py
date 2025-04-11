from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.kernel.bitfield_attention import (
    _bitfield_attn_backward,
    _bitfield_attn_forward,
)


def call_bitfield_attention_forward(*args, **kwargs):
    return _bitfield_attn_forward(*args, **kwargs)


class ContextParallelBitfieldAttention(torch.autograd.Function):

    stream: torch.cuda.Stream = None

    @torch.cuda.nvtx.range("ContextParallelBitfieldAttention.forward")
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bitfield_mask: torch.Tensor,
        compressed_mask: torch.Tensor,
        offsets_per_rank: list[torch.Tensor],
        sp_group: dist.ProcessGroup,
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
        if ContextParallelBitfieldAttention.stream is None:
            ContextParallelBitfieldAttention.stream = torch.cuda.Stream()

        stream = ContextParallelBitfieldAttention.stream

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, nheads_kv, _ = k.shape

        seqlen_per_rank = [len(offsets) for offsets in offsets_per_rank]
        total_seqlen = sum(seqlen_per_rank)
        offsets_q = offsets_per_rank[dist.get_rank(sp_group)]
        offsets_kv = torch.cat(offsets_per_rank, dim=0)

        assert bitfield_mask.shape == (batch, total_seqlen)
        # assert k.shape == (batch, seqlen_kv, nheads, d)
        # assert v.shape == (batch, seqlen_kv, nheads, d)
        assert (
            nheads_kv % heads_stride == 0
        ), "number of heads must be divisible by heads_stride"
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda

        per_head_events: list[torch.cuda.Event] = []

        # pre-allocate memory for k and v gathering for all ranks
        gathered_kv = [
            torch.empty(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )
            for _ in range(nheads_kv // heads_stride)
        ]

        # Initialize allgather works asynchronously
        with torch.cuda.stream(stream):
            for head_index in range(0, nheads_kv, heads_stride):
                with torch.cuda.nvtx.range(f"all_gather_{head_index}"):
                    work = dist.all_gather(
                        list(
                            gathered_kv[head_index // heads_stride].split(
                                seqlen_per_rank, dim=2
                            )
                        ),
                        torch.stack(
                            [
                                k[:, :, head_index : head_index + heads_stride, :],
                                v[:, :, head_index : head_index + heads_stride, :],
                            ],
                            dim=0,
                        ).contiguous(),
                        group=sp_group,
                        async_op=True,
                    )
                    work.wait()

                    event = torch.cuda.Event()
                    event.record()
                    per_head_events.append(event)

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        softmax_scales: list[torch.Tensor] = []

        assert len(per_head_events) == nheads_kv // heads_stride
        group_size = nheads // nheads_kv
        for head_index in range(0, nheads_kv, heads_stride):
            event = per_head_events[head_index // heads_stride]
            torch.cuda.current_stream().wait_event(event)

            current_q = q[
                :, :, head_index * group_size : (head_index + heads_stride) * group_size
            ].contiguous()
            current_k = gathered_kv[head_index // heads_stride][0]
            current_v = gathered_kv[head_index // heads_stride][1]

            o, lse, softmax_scale_out = call_bitfield_attention_forward(
                current_q,
                current_k,
                current_v,
                bitfield_mask,
                compressed_mask,
                offsets_q,
                offsets_kv,
                bias=bias,
                softmax_scale=softmax_scale,
            )

            os.append(o)
            lses.append(lse)
            softmax_scales.append(softmax_scale_out)

        ctx.save_for_backward(q, k, v, bias, bitfield_mask, compressed_mask)
        ctx.offsets_per_rank = offsets_per_rank
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scales = softmax_scales
        ctx.sp_group = sp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        stream = ContextParallelBitfieldAttention.stream
        (q, k, v, bias, bitfield_mask, compressed_mask) = ctx.saved_tensors
        offsets_per_rank: list[torch.Tensor] = ctx.offsets_per_rank
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scales: list[torch.Tensor] = ctx.softmax_scales
        sp_group: dist.ProcessGroup = ctx.sp_group

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, nheads_kv, _ = k.shape

        seqlen_per_rank = [len(offsets) for offsets in offsets_per_rank]
        total_seqlen = sum(seqlen_per_rank)
        offsets_q = offsets_per_rank[dist.get_rank(sp_group)]
        offsets_kv = torch.cat(offsets_per_rank, dim=0)

        per_head_events: list[torch.cuda.Event] = []

        # pre-allocate memory for k and v gathering for all heads
        gathered_kv = [
            torch.empty(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )
            for _ in range(nheads_kv // heads_stride)
        ]

        # Initialize allgather works asynchronously
        with torch.cuda.stream(stream):
            for head_index in range(0, nheads_kv, heads_stride):
                dist.all_gather(
                    list(
                        gathered_kv[head_index // heads_stride][0].split(
                            seqlen_per_rank, dim=1
                        )
                    ),
                    k[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                dist.all_gather(
                    list(
                        gathered_kv[head_index // heads_stride][1].split(
                            seqlen_per_rank, dim=1
                        )
                    ),
                    v[:, :, head_index : head_index + heads_stride, :].contiguous(),
                    group=sp_group,
                    async_op=True,
                )
                event = torch.cuda.Event()
                event.record(stream)
                per_head_events.append(event)

        dqs: list[torch.Tensor] = []

        assert len(per_head_events) == nheads_kv // heads_stride
        group_size = nheads // nheads_kv
        scattered_dkv = torch.zeros(
            (2, batch, seqlen_kv, nheads_kv, d),
            dtype=k.dtype,
            device=k.device,
        )
        for head_index in range(0, nheads_kv, heads_stride):
            kv_head_index = head_index // heads_stride
            event = per_head_events[kv_head_index]
            torch.cuda.current_stream().wait_event(event)

            dq = torch.empty(
                (batch, seqlen_q, heads_stride * group_size, d),
                dtype=q.dtype,
                device=q.device,
            )
            dkv = torch.empty(
                (2, batch, seqlen_kv, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )

            dgkv = torch.zeros(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=torch.float32,
                device=k.device,
            )

            _bitfield_attn_backward(
                do=do[
                    :,
                    :,
                    head_index * group_size : (head_index + heads_stride) * group_size,
                ],
                q=q[
                    :,
                    :,
                    head_index * group_size : (head_index + heads_stride) * group_size,
                ],
                k=gathered_kv[kv_head_index][0],
                v=gathered_kv[kv_head_index][1],
                o=os[head_index // heads_stride],
                bitfield_mask=bitfield_mask,
                compressed_mask=compressed_mask,
                lse=lses[head_index // heads_stride],
                dq=dq,
                dk=dgkv[0],
                dv=dgkv[1],
                offsets_q=offsets_q,
                offsets_k=offsets_kv,
                bias=bias,
                softmax_scale=softmax_scales[head_index // heads_stride],
            )

            with torch.cuda.stream(stream):
                dist.reduce_scatter(
                    dkv[0],
                    list(dgkv[0].to(dtype=dkv.dtype).split(seqlen_per_rank, dim=1)),
                    group=sp_group,
                    async_op=True,
                )
                dist.reduce_scatter(
                    dkv[1],
                    list(dgkv[1].to(dtype=dkv.dtype).split(seqlen_per_rank, dim=1)),
                    group=sp_group,
                    async_op=True,
                )

            dqs.append(dq)
            scattered_dkv[0][:, :, kv_head_index : kv_head_index + heads_stride] += dkv[
                0
            ]
            scattered_dkv[1][:, :, kv_head_index : kv_head_index + heads_stride] += dkv[
                1
            ]

        torch.cuda.current_stream().wait_stream(stream)

        return (
            torch.cat(dqs, dim=2),
            scattered_dkv[0],
            scattered_dkv[1],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def context_parallel_bitfield_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bitfield_mask: torch.Tensor,
    compressed_mask: torch.Tensor,
    offsets_per_rank: list[torch.Tensor],
    sp_group: dist.ProcessGroup,
    heads_stride: int = 1,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert (
        bitfield_mask is not None and bitfield_mask.dtype == torch.int64
    ), "Bitfield attention requires an attention mask of type torch.int64."

    # BAM follows FA2 that uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelBitfieldAttention.apply(
        query,
        key,
        value,
        bitfield_mask,
        compressed_mask,
        offsets_per_rank,
        sp_group,
        None,
        None,
        heads_stride,
    )

    return attn_output, None
