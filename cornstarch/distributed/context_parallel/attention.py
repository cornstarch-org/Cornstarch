"""Context-parallel all-gather flash attention.

Each CP rank holds a subsequence of Q, K, V.  Before computing attention,
each rank gathers the full K and V from all CP ranks so that every rank
attends over the complete key-value sequence.  The all-gather is dispatched
per head group into a side CUDA stream to overlap communication with
computation.

On the backward pass, ``dk``/``dv`` gradients are reduce-scattered back so
each rank accumulates only the gradient for its local K/V slice.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward


def _allgather_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    nheads: int,
    heads_stride: int,
    seqlen_per_rank: list[int],
    total_seqlen: int,
    cp_group: dist.ProcessGroup,
    stream: torch.cuda.Stream,
) -> tuple[list[torch.Tensor], list[torch.cuda.Event]]:
    """All-gather K and V per head group on a side stream.

    Returns ``(gathered_kv, events)`` where ``gathered_kv[gi]`` is a
    ``(2, batch, total_seqlen, heads_stride, d)`` tensor and ``events[gi]``
    signals when that head group's gather is complete.
    """
    batch, _, _, d = k.shape
    gathered_kv = [
        torch.empty(
            (2, batch, total_seqlen, heads_stride, d),
            dtype=k.dtype,
            device=k.device,
        )
        for _ in range(nheads // heads_stride)
    ]

    events: list[torch.cuda.Event] = []
    with torch.cuda.stream(stream):
        for hi in range(0, nheads, heads_stride):
            gi = hi // heads_stride
            dist.all_gather(
                list(gathered_kv[gi][0].split(seqlen_per_rank, dim=1)),
                k[:, :, hi : hi + heads_stride, :].contiguous(),
                group=cp_group,
                async_op=True,
            )
            dist.all_gather(
                list(gathered_kv[gi][1].split(seqlen_per_rank, dim=1)),
                v[:, :, hi : hi + heads_stride, :].contiguous(),
                group=cp_group,
                async_op=True,
            )
            evt = torch.cuda.Event()
            evt.record(stream)
            events.append(evt)

    return gathered_kv, events


class ContextParallelFlashAttention(torch.autograd.Function):
    """Custom autograd function that all-gathers K/V before flash attention.

    Q stays local; K and V are gathered across the CP group so every rank
    attends over the full key-value sequence.  The gather is split by head
    groups (``heads_stride`` heads per call) and overlapped with compute via
    a dedicated CUDA stream.  On the backward pass, full-sequence ``dk``/``dv``
    are computed against the gathered K/V and then reduce-scattered back to
    each rank's local slice.
    """

    _stream: torch.cuda.Stream | None = None

    @staticmethod
    def _get_stream() -> torch.cuda.Stream:
        if ContextParallelFlashAttention._stream is None:
            ContextParallelFlashAttention._stream = torch.cuda.Stream()
        return ContextParallelFlashAttention._stream

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cp_group: dist.ProcessGroup,
        heads_stride: int = 1,
    ) -> torch.Tensor:
        """Run flash attention with all-gathered K/V.

        Tensors are in ``(batch, seq, heads, dim)`` layout.  ``heads_stride``
        controls how many heads are gathered together per collective call.
        """
        stream = ContextParallelFlashAttention._get_stream()
        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape

        assert k.shape == (batch, seqlen_kv, nheads, d)
        assert v.shape == (batch, seqlen_kv, nheads, d)
        assert nheads % heads_stride == 0
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert q.is_cuda and k.is_cuda and v.is_cuda

        cp_size = dist.get_world_size(cp_group)
        seqlen_per_rank_t = [
            torch.empty(1, dtype=torch.long, device=k.device) for _ in range(cp_size)
        ]
        dist.all_gather(
            seqlen_per_rank_t,
            torch.tensor(seqlen_kv, device=k.device),
            group=cp_group,
        )
        seqlen_per_rank = [int(t.item()) for t in seqlen_per_rank_t]
        total_seqlen = sum(seqlen_per_rank)

        gathered_kv, per_head_events = _allgather_kv(
            k, v, nheads, heads_stride, seqlen_per_rank, total_seqlen,
            cp_group, stream,
        )

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        softmax_scale = d ** (-0.5)

        for hi in range(0, nheads, heads_stride):
            gi = hi // heads_stride
            torch.cuda.current_stream().wait_event(per_head_events[gi])

            o, lse, _, _ = _flash_attn_forward(
                q[:, :, hi : hi + heads_stride, :].contiguous(),
                gathered_kv[gi][0],
                gathered_kv[gi][1],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
            os.append(o)
            lses.append(lse)

        ctx.save_for_backward(q, k, v)
        ctx.seqlen_per_rank = seqlen_per_rank
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        stream = ContextParallelFlashAttention._get_stream()

        q, k, v = ctx.saved_tensors
        seqlen_per_rank: list[int] = ctx.seqlen_per_rank
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scale: float = ctx.softmax_scale
        cp_group: dist.ProcessGroup = ctx.cp_group

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape
        total_seqlen = sum(seqlen_per_rank)

        gathered_kv, per_head_events = _allgather_kv(
            k, v, nheads, heads_stride, seqlen_per_rank, total_seqlen,
            cp_group, stream,
        )

        dqs: list[torch.Tensor] = []
        dks: list[torch.Tensor] = []
        dvs: list[torch.Tensor] = []

        for hi in range(0, nheads, heads_stride):
            gi = hi // heads_stride
            torch.cuda.current_stream().wait_event(per_head_events[gi])

            dq = torch.empty((batch, seqlen_q, heads_stride, d), dtype=q.dtype, device=q.device)
            dkv = torch.empty((2, batch, seqlen_kv, heads_stride, d), dtype=k.dtype, device=k.device)
            dgkv = torch.zeros((2, batch, total_seqlen, heads_stride, d), dtype=k.dtype, device=k.device)

            _flash_attn_backward(
                dout=do[:, :, hi : hi + heads_stride, :],
                q=q[:, :, hi : hi + heads_stride, :],
                k=gathered_kv[gi][0],
                v=gathered_kv[gi][1],
                out=os[gi],
                softmax_lse=lses[gi],
                dq=dq,
                dk=dgkv[0],
                dv=dgkv[1],
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
            )

            with torch.cuda.stream(stream):
                dist.reduce_scatter(
                    dkv[0],
                    list(dgkv[0].split(seqlen_per_rank, dim=1)),
                    group=cp_group,
                    async_op=True,
                )
                dist.reduce_scatter(
                    dkv[1],
                    list(dgkv[1].split(seqlen_per_rank, dim=1)),
                    group=cp_group,
                    async_op=True,
                )

            dqs.append(dq.clone())
            dks.append(dkv[0].clone())
            dvs.append(dkv[1].clone())

        torch.cuda.current_stream().wait_stream(stream)

        return (
            torch.cat(dqs, dim=2),
            torch.cat(dks, dim=2),
            torch.cat(dvs, dim=2),
            None,
            None,
        )


def context_parallel_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_group: dist.ProcessGroup,
    heads_stride: int = 1,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transpose from HF ``(b, h, s, d)`` to flash-attn ``(b, s, h, d)``, run CP attention, transpose back."""
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelFlashAttention.apply(
        query, key, value, cp_group, heads_stride
    )
    return attn_output.transpose(1, 2), None
