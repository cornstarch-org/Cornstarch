import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward


class ContextParallelFlashAttention(torch.autograd.Function):

    stream: torch.cuda.Stream = None

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        heads_stride: int = 1,
    ) -> torch.Tensor:
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_kv, nheads, headdim)
        mask: (batch_size, seqlen_q, total_seqlen)
            columns should be aligned with the order of gathered kv
        seqlen_per_rank: (world_size(sp_group),)
        """
        if ContextParallelFlashAttention.stream is None:
            ContextParallelFlashAttention.stream = torch.cuda.Stream()

        stream = ContextParallelFlashAttention.stream

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape

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
        seqlen_per_rank = [
            torch.empty(1, dtype=torch.long, device=k.device)
            for _ in range(dist.get_world_size(sp_group))
        ]
        dist.all_gather(
            seqlen_per_rank, torch.tensor(seqlen_kv, device=k.device), group=sp_group
        )

        seqlen_per_rank = [seqlen.item() for seqlen in seqlen_per_rank]
        total_seqlen = sum(seqlen_per_rank)
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

        os: list[torch.Tensor] = []
        lses: list[torch.Tensor] = []

        softmax_scale = q.shape[-1] ** (-0.5)
        assert len(per_head_events) == nheads // heads_stride
        for head_index in range(0, nheads, heads_stride):
            event = per_head_events[head_index // heads_stride]
            torch.cuda.current_stream().wait_event(event)

            current_q = q[:, :, head_index : head_index + heads_stride, :].contiguous()
            current_k = gathered_kv[head_index // heads_stride][0]
            current_v = gathered_kv[head_index // heads_stride][1]
            assert current_k.is_contiguous() and current_v.is_contiguous()

            o, lse, _, _ = _flash_attn_forward(
                current_q,
                current_k,
                current_v,
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
        ctx.sp_group = sp_group

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        stream = ContextParallelFlashAttention.stream

        q, k, v = ctx.saved_tensors
        seqlen_per_rank: list[int] = ctx.seqlen_per_rank
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scale: float = ctx.softmax_scale
        sp_group: dist.ProcessGroup = ctx.sp_group

        batch, seqlen_q, nheads, d = q.shape
        _, seqlen_kv, _, _ = k.shape

        per_head_events: list[torch.cuda.Event] = []

        # pre-allocate memory for k and v gathering for all heads
        total_seqlen = sum(seqlen_per_rank)
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

            dgkv = torch.zeros(
                (2, batch, total_seqlen, heads_stride, d),
                dtype=k.dtype,
                device=k.device,
            )

            _flash_attn_backward(
                dout=do[:, :, head_index : head_index + heads_stride, :],
                q=q[:, :, head_index : head_index + heads_stride, :],
                k=gathered_kv[head_index // heads_stride][0],
                v=gathered_kv[head_index // heads_stride][1],
                out=os[head_index // heads_stride],
                softmax_lse=lses[head_index // heads_stride],
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
                    group=sp_group,
                    async_op=True,
                )
                dist.reduce_scatter(
                    dkv[1],
                    list(dgkv[1].split(seqlen_per_rank, dim=1)),
                    group=sp_group,
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
            None,
        )


def context_parallel_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sp_group: dist.ProcessGroup,
    heads_stride: int = 1,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # FA1 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelFlashAttention.apply(
        query, key, value, sp_group, heads_stride
    )

    return attn_output, None
