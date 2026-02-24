import torch
import torch.distributed as dist
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)


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


class ContextParallelFlashAttentionVarlen(torch.autograd.Function):
    """Context-parallel flash attention for variable-length sequences.

    q, k, v: [total_local, nheads, headdim] — flat varlen format (NOT batched).
    cu_seqlens_q: [N+1] int32 — cumulative local chunk lengths.
    cu_seqlens_k_global: [N+1] int32 — cumulative full image lengths.

    Forward: all-gather K/V from all SP ranks (rank-interleaved), reorder to
    image-contiguous layout, then call flash_attn_varlen_fwd.
    Backward: recompute reordered K/V, run flash_attn_varlen_bwd, all-reduce
    dK/dV across ranks, and slice to recover each rank's local gradient.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_global: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        total_local, nheads, d = q.shape
        p = dist.get_world_size(sp_group)
        my_rank = dist.get_rank(sp_group)
        softmax_scale = d ** (-0.5)

        # 1. All-gather seqlen_per_rank
        seqlen_tensors = [
            torch.empty(1, dtype=torch.long, device=k.device) for _ in range(p)
        ]
        dist.all_gather(
            seqlen_tensors,
            torch.tensor(total_local, device=k.device),
            group=sp_group,
        )
        seqlen_per_rank = [s.item() for s in seqlen_tensors]
        total_global = sum(seqlen_per_rank)

        cum_spr = [0] * (p + 1)
        for r in range(p):
            cum_spr[r + 1] = cum_spr[r] + seqlen_per_rank[r]

        # 2. Compute inv_perm: maps reordered_pos -> gathered_pos.
        # After all-gather, gathered KV is rank-interleaved:
        #   [rank0_chunks, rank1_chunks, ..., rank_{p-1}_chunks]
        # Each rank r holds: [chunk_r_img_0, ..., chunk_r_img_{N-1}]
        # chunk_r_img_i has n_ri = seqlens[i]*(r+1)//p - seqlens[i]*r//p tokens.
        # Desired image-contiguous order: [img_0_full, img_1_full, ...]
        # where img_i_full = [chunk_0_img_i, ..., chunk_{p-1}_img_i].
        seqlens = (cu_seqlens_k_global[1:] - cu_seqlens_k_global[:-1]).cpu().long()
        N = seqlens.shape[0]
        local_n = torch.stack(
            [seqlens * (r + 1) // p - seqlens * r // p for r in range(p)], dim=0
        )  # [p, N]
        cu_k_cpu = cu_seqlens_k_global.cpu().long()

        inv_perm = torch.empty(total_global, dtype=torch.long)
        for r in range(p):
            offset_r = cum_spr[r]
            cum_img_r = 0
            for i in range(N):
                n_ri = local_n[r, i].item()
                if n_ri > 0:
                    gathered_start = offset_r + cum_img_r
                    reordered_start = cu_k_cpu[i].item() + local_n[:r, i].sum().item()
                    inv_perm[reordered_start : reordered_start + n_ri] = torch.arange(
                        gathered_start, gathered_start + n_ri, dtype=torch.long
                    )
                cum_img_r += n_ri

        inv_perm = inv_perm.to(k.device)

        # 3. All-gather K and V
        gathered_k = torch.empty(total_global, nheads, d, dtype=k.dtype, device=k.device)
        gathered_v = torch.empty(total_global, nheads, d, dtype=v.dtype, device=v.device)
        dist.all_gather(
            [gathered_k[cum_spr[r] : cum_spr[r + 1]] for r in range(p)],
            k.contiguous(),
            group=sp_group,
        )
        dist.all_gather(
            [gathered_v[cum_spr[r] : cum_spr[r + 1]] for r in range(p)],
            v.contiguous(),
            group=sp_group,
        )

        # 4. Reorder gathered KV to image-contiguous layout
        reordered_k = gathered_k[inv_perm].contiguous()
        reordered_v = gathered_v[inv_perm].contiguous()

        # 5. Flash attention varlen forward
        out, softmax_lse, _, _ = _flash_attn_varlen_forward(
            q.contiguous(),
            reordered_k,
            reordered_v,
            cu_seqlens_q,
            cu_seqlens_k_global,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        ctx.save_for_backward(
            q, k, v, reordered_k, reordered_v, out, softmax_lse, inv_perm
        )
        ctx.seqlen_per_rank = seqlen_per_rank
        ctx.cum_spr = cum_spr
        ctx.softmax_scale = softmax_scale
        ctx.sp_group = sp_group
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k_global = cu_seqlens_k_global
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.my_rank = my_rank

        return out

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, do: torch.Tensor):
        (
            q,
            k,
            v,
            reordered_k,
            reordered_v,
            out,
            softmax_lse,
            inv_perm,
        ) = ctx.saved_tensors
        seqlen_per_rank: list[int] = ctx.seqlen_per_rank
        cum_spr: list[int] = ctx.cum_spr
        softmax_scale: float = ctx.softmax_scale
        sp_group: dist.ProcessGroup = ctx.sp_group
        cu_seqlens_q = ctx.cu_seqlens_q
        cu_seqlens_k_global = ctx.cu_seqlens_k_global
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        my_rank = ctx.my_rank

        total_local, nheads, d = q.shape
        total_global = sum(seqlen_per_rank)

        dq = torch.empty_like(q)
        dk_reordered = torch.zeros(
            total_global, nheads, d, dtype=k.dtype, device=k.device
        )
        dv_reordered = torch.zeros(
            total_global, nheads, d, dtype=v.dtype, device=v.device
        )

        _flash_attn_varlen_backward(
            dout=do.contiguous(),
            q=q.contiguous(),
            k=reordered_k,
            v=reordered_v,
            out=out,
            softmax_lse=softmax_lse,
            dq=dq,
            dk=dk_reordered,
            dv=dv_reordered,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k_global,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
        )

        # Undo reorder: since reordered_k = gathered_k[inv_perm],
        # the backward gives dk_gathered = dk_reordered[argsort(inv_perm)].
        perm = torch.argsort(inv_perm)
        dk_gathered = dk_reordered[perm].contiguous()  # [total_global, nheads, d]
        dv_gathered = dv_reordered[perm].contiguous()

        # Sum contributions from all SP ranks (each rank computed partial gradients
        # from its own Q chunks attending to the full global KV).
        dist.all_reduce(dk_gathered, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(dv_gathered, op=dist.ReduceOp.SUM, group=sp_group)

        dk = dk_gathered[cum_spr[my_rank] : cum_spr[my_rank + 1]].contiguous()
        dv = dv_gathered[cum_spr[my_rank] : cum_spr[my_rank + 1]].contiguous()

        return dq, dk, dv, None, None, None, None, None


def context_parallel_varlen_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sp_group: dist.ProcessGroup,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_global: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Context-parallel flash attention for variable-length sequences.

    q, k, v: [total_local, nheads, headdim] — flat varlen format.
    cu_seqlens_q: local chunk cumulative lengths (int32).
    cu_seqlens_k_global: full image cumulative lengths (int32).
    """
    return ContextParallelFlashAttentionVarlen.apply(
        q,
        k,
        v,
        sp_group,
        cu_seqlens_q,
        cu_seqlens_k_global,
        max_seqlen_q,
        max_seqlen_k,
    )
