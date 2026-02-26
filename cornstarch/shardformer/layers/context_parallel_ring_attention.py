"""
Context-parallel ring attention for packed (varlen) sequences with zigzag partitioning.

Each SP rank holds two non-contiguous chunks of every sequence (zigzag layout).
This module all-gathers K/V from all ranks, reorders from rank-interleaved to global
chunk order, and calls the Triton flash_attn_varlen_func with per-sequence
q_seq_offsets for offset-aware causal masking.

Communication is pipelined: the next heads_stride all-gather is started on a
background CUDA stream while computation runs on the main stream.

Variable-size all_gather:
    Ranks may hold different numbers of packed tokens.  We pad K/V to
    max_local_k before all_gather and slice back to actual sizes after.
    We also exchange chunk_a token counts per rank so that _build_zigzag_inv_perm
    can correctly determine the chunk_a / chunk_b boundary within each rank's
    gathered data.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from cornstarch.kernel.attention import flash_attn_varlen_func, _flash_attn_varlen_backward


def _build_deinterleave_perm(
    sp_size: int,
    B: int,
    cseq_all: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a permutation that maps chunk-major-sample-interleaved order (the output
    of _build_zigzag_inv_perm) to sample-major order required by cu_seqlens_k_global.

    After all_gather + inv_perm, the gathered K has structure:
        [chunk0_s0, chunk0_s1, ..., chunk1_s0, chunk1_s1, ..., chunk_{2P-1}_s{B-1}]

    We need sample-major order:
        [s0_all_chunks, s1_all_chunks, ..., s{B-1}_all_chunks]

    Parameters
    ----------
    sp_size   : int — number of SP ranks.
    B         : int — batch size (number of sequences).
    cseq_all  : list of sp_size tensors, each (2B+1,) int32 — cu_seqlens_q per rank.
    device    : torch.device.

    Returns
    -------
    deinterleave_perm : [total_global_k] LongTensor
    """
    total_chunks = 2 * sp_size

    # Per-chunk per-sample sizes
    # Global chunk c is held by:
    #   c < sp_size  : chunk_a of rank c  → sizes from cseq_all[c][b+1] - cseq_all[c][b]
    #   c >= sp_size : chunk_b of rank (2P-1-c) → sizes from cseq_all[r][B+b+1]-cseq_all[r][B+b]
    sizes: list[list[int]] = []   # sizes[c][b]
    for c in range(total_chunks):
        if c < sp_size:
            r = c
            chunk_sizes_c = [int((cseq_all[r][b + 1] - cseq_all[r][b]).item())
                             for b in range(B)]
        else:
            r = total_chunks - 1 - c
            chunk_sizes_c = [int((cseq_all[r][B + b + 1] - cseq_all[r][B + b]).item())
                             for b in range(B)]
        sizes.append(chunk_sizes_c)

    # Cumulative positions in chunk-major order (= input positions after inv_perm)
    cum_chunk: list[int] = [0] * (total_chunks + 1)
    for c in range(total_chunks):
        cum_chunk[c + 1] = cum_chunk[c] + sum(sizes[c])
    total_global_k = cum_chunk[total_chunks]

    # Per-sample total sizes and cumulative offsets in sample-major order (= output)
    vl = [sum(sizes[c][b] for c in range(total_chunks)) for b in range(B)]
    cum_sample: list[int] = [0] * (B + 1)
    for b in range(B):
        cum_sample[b + 1] = cum_sample[b] + vl[b]

    # Cumulative per-sample-per-chunk offsets within each sample
    cum_ws: list[list[int]] = [[0] * (total_chunks + 1) for _ in range(B)]
    for b in range(B):
        for c in range(total_chunks):
            cum_ws[b][c + 1] = cum_ws[b][c] + sizes[c][b]

    # Build permutation: deinterleave_perm[output_pos] = input_pos
    deinterleave_perm = torch.empty(total_global_k, dtype=torch.long, device=device)
    for c in range(total_chunks):
        cum_in_chunk = 0
        for b in range(B):
            csz = sizes[c][b]
            if csz == 0:
                continue
            inp_start = cum_chunk[c] + cum_in_chunk
            out_start = cum_sample[b] + cum_ws[b][c]
            deinterleave_perm[out_start: out_start + csz] = torch.arange(
                inp_start, inp_start + csz, device=device
            )
            cum_in_chunk += csz

    return deinterleave_perm


def _build_zigzag_inv_perm(
    seqlen_per_rank: list[int],
    chunk_a_per_rank: list[int],
    sp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build inv_perm that maps rank-interleaved gathered order → global chunk order.

    After all_gather (and slicing padding), the layout is:
        [rank0_tokens, rank1_tokens, ..., rank_{P-1}_tokens]

    Within rank r's block:
        positions  [0,            chunk_a_per_rank[r])  → chunk r   tokens
        positions  [chunk_a_per_rank[r], seqlen_per_rank[r]) → chunk 2P-1-r tokens

    We want global chunk order:
        [chunk_0_tokens, chunk_1_tokens, ..., chunk_{2P-1}_tokens]

    Parameters
    ----------
    seqlen_per_rank  : list[int], length sp_size — total packed tokens per rank.
    chunk_a_per_rank : list[int], length sp_size — packed chunk_a tokens per rank
                       (= cu_seqlens_q[B] on that rank).
    sp_size          : int — number of SP ranks.
    device           : torch.device.

    Returns
    -------
    inv_perm : [total_global_k] LongTensor  —  gathered_pos[inv_perm[g]] == global_pos[g]
    """
    p = sp_size
    chunk_b_per_rank = [seqlen_per_rank[r] - chunk_a_per_rank[r] for r in range(p)]

    # Cumulative rank offsets in the rank-interleaved gathered tensor
    cum_rank = [0] * (p + 1)
    for r in range(p):
        cum_rank[r + 1] = cum_rank[r] + seqlen_per_rank[r]

    total_global = cum_rank[p]
    inv_perm = torch.empty(total_global, dtype=torch.long, device=device)

    global_pos = 0
    for c in range(2 * p):
        # Rank that holds chunk c
        r = min(c, 2 * p - 1 - c)
        # Within rank r, chunk c is chunk_a if c == r, else chunk_b
        is_b = (c != r)
        if not is_b:
            chunk_len = chunk_a_per_rank[r]
            src_start = cum_rank[r]
        else:
            chunk_len = chunk_b_per_rank[r]
            src_start = cum_rank[r] + chunk_a_per_rank[r]
        inv_perm[global_pos: global_pos + chunk_len] = torch.arange(
            src_start, src_start + chunk_len, device=device
        )
        global_pos += chunk_len

    return inv_perm


class ContextParallelVarlenRingAttention(torch.autograd.Function):
    """
    Context-parallel ring attention for packed (varlen) Q/K/V tensors.

    cu_seqlens_q is expected to have 2B+1 entries (2B = 2 * batch_size sub-sequences):
    the first B entries correspond to chunk_a sub-sequences (in sample order) and
    the next B entries to chunk_b sub-sequences.  cu_seqlens_q[B] is therefore the
    total packed chunk_a token count on the local rank and is used to tell each
    remote rank how to split its gathered data into chunk_a / chunk_b.

    All K/V all_gather calls are padded to max_local_k so that all ranks contribute
    tensors of the same shape to dist.all_gather.  Gradients are reduce_scattered
    with the same padding scheme.
    """

    stream: torch.cuda.Stream = None

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_global: torch.Tensor,
        q_seq_offsets: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        heads_stride: int = 1,
    ) -> torch.Tensor:
        if ContextParallelVarlenRingAttention.stream is None:
            ContextParallelVarlenRingAttention.stream = torch.cuda.Stream()
        stream = ContextParallelVarlenRingAttention.stream

        total_local_q, nheads,    headdim = q.shape
        total_local_k, nheads_kv, _       = k.shape
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        assert nheads % nheads_kv == 0, "nheads must be divisible by nheads_kv (GQA)"
        assert nheads_kv % heads_stride == 0
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in (torch.float16, torch.bfloat16)
        assert q.is_cuda and k.is_cuda and v.is_cuda

        n_seqs = cu_seqlens_q.shape[0] - 1   # = 2B
        B      = n_seqs // 2
        # Number of packed tokens from chunk_a sub-sequences on local rank
        chunk_a_tokens_local = int(cu_seqlens_q[B].item())

        # ── 1. Exchange seqlen_per_rank, chunk_a_per_rank and cu_seqlens_q ─
        meta = [torch.empty(2, dtype=torch.long, device=k.device) for _ in range(sp_size)]
        dist.all_gather(
            meta,
            torch.tensor([total_local_k, chunk_a_tokens_local], device=k.device),
            group=sp_group,
        )
        seqlen_per_rank  = [int(m[0].item()) for m in meta]
        chunk_a_per_rank = [int(m[1].item()) for m in meta]
        total_global_k   = sum(seqlen_per_rank)
        max_local_k      = max(seqlen_per_rank)

        # All-gather cu_seqlens_q from all ranks to reconstruct per-rank per-sample
        # chunk sizes for the de-interleave permutation (needed for B > 1).
        cseq_all = [torch.empty_like(cu_seqlens_q) for _ in range(sp_size)]
        dist.all_gather(cseq_all, cu_seqlens_q, group=sp_group)

        # ── 2. Build reorder permutations ─────────────────────────────────
        inv_perm = _build_zigzag_inv_perm(
            seqlen_per_rank, chunk_a_per_rank, sp_size, k.device
        )
        # De-interleave perm: chunk-major-sample-interleaved → sample-major
        # (required when B > 1 so that each Q sub-sequence sees only its own
        # sample's K tokens, as expected by cu_seqlens_k_global).
        deinterleave_perm = _build_deinterleave_perm(sp_size, B, cseq_all, k.device)
        inv_deinterleave_perm = torch.argsort(deinterleave_perm)

        # ── 3. Helpers for send/receive padding ───────────────────────────
        def _make_kv_send(k_h: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
            """Stack K/V for current head stride and pad token dim to max_local_k."""
            kv = torch.stack([k_h.contiguous(), v_h.contiguous()], dim=0)
            if total_local_k < max_local_k:
                kv = F.pad(kv, (0, 0, 0, 0, 0, max_local_k - total_local_k))
            return kv

        def _unpad_and_cat(bufs: list[torch.Tensor], idx: int) -> torch.Tensor:
            """Concatenate rank-sliced actual-size tensors from kv_buffers."""
            return torch.cat(
                [bufs[r][idx, :seqlen_per_rank[r]] for r in range(sp_size)], dim=0
            )

        # ── 4. Launch all all-gathers upfront with separate per-head buffers ─
        # Each head stride gets its own buffer so there's no aliasing between
        # the background stream's writes and the main stream's reads.
        # After each all_gather we record a CUDA event on the background stream
        # and wait_event on the main stream before using the data.  This is the
        # same pattern used by ContextParallelFlashAttention and correctly
        # synchronises gloo H2D copies (which are enqueued on the background
        # CUDA stream but must be visible on the main stream).
        n_head_groups = nheads_kv // heads_stride
        per_head_bufs: list[list[torch.Tensor]] = []
        per_head_events: list[torch.cuda.Event] = []

        with torch.cuda.stream(stream):
            for hi in range(0, nheads_kv, heads_stride):
                buf = [
                    torch.empty(
                        (2, max_local_k, heads_stride, headdim),
                        dtype=k.dtype, device=k.device,
                    )
                    for _ in range(sp_size)
                ]
                dist.all_gather(
                    buf,
                    _make_kv_send(k[:, hi: hi + heads_stride],
                                  v[:, hi: hi + heads_stride]),
                    group=sp_group,
                    async_op=True,
                )
                ev = torch.cuda.Event()
                ev.record(stream)
                per_head_bufs.append(buf)
                per_head_events.append(ev)

        os:   list[torch.Tensor] = []
        lses: list[torch.Tensor] = []
        group_size    = nheads // nheads_kv
        softmax_scale = headdim ** (-0.5)

        from cornstarch.kernel.attention import _flash_attn_varlen_forward
        for head_index in range(0, nheads_kv, heads_stride):
            gi = head_index // heads_stride
            torch.cuda.current_stream().wait_event(per_head_events[gi])

            kv_buf = per_head_bufs[gi]

            # Slice to actual sizes, concatenate, reorder to global chunk order.
            # Duplicate K/V so that the 2B sub-sequences in cu_seqlens_k_global
            # can each reference their own copy of the global K (chunk_a seqs use
            # the first half [0, total_global_k), chunk_b seqs use the second half
            # [total_global_k, 2*total_global_k)).
            # 1) inv_perm: rank-interleaved → chunk-major-sample-interleaved
            # 2) deinterleave_perm: chunk-major-sample-interleaved → sample-major
            # 3) Duplicate so chunk_a seqs use [0, total_global_k) and chunk_b
            #    seqs use [total_global_k, 2*total_global_k) of the global K,
            #    matching cu_seqlens_k_global.
            gathered_k_half = _unpad_and_cat(kv_buf, 0)[inv_perm][deinterleave_perm]
            gathered_v_half = _unpad_and_cat(kv_buf, 1)[inv_perm][deinterleave_perm]
            gathered_k = torch.cat([gathered_k_half, gathered_k_half], dim=0).contiguous()
            gathered_v = torch.cat([gathered_v_half, gathered_v_half], dim=0).contiguous()

            q_h_start = head_index * group_size
            q_h_end   = (head_index + heads_stride) * group_size
            q_slice   = q[:, q_h_start:q_h_end, :].contiguous()

            o, lse, _ = _flash_attn_varlen_forward(
                q_slice, gathered_k, gathered_v,
                cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets,
                max_seqlen_q, max_seqlen_k,
                softmax_scale,
            )
            os.append(o)
            lses.append(lse)

        torch.cuda.current_stream().wait_stream(stream)

        ctx.save_for_backward(q, k, v, inv_perm, deinterleave_perm, inv_deinterleave_perm,
                               cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets)
        ctx.seqlen_per_rank  = seqlen_per_rank
        ctx.chunk_a_per_rank = chunk_a_per_rank
        ctx.max_local_k      = max_local_k
        ctx.total_local_k    = total_local_k
        ctx.heads_stride     = heads_stride
        ctx.os               = os
        ctx.lses             = lses
        ctx.softmax_scale    = softmax_scale
        ctx.sp_group         = sp_group
        ctx.max_seqlen_q     = max_seqlen_q
        ctx.max_seqlen_k     = max_seqlen_k
        ctx.group_size       = group_size
        ctx.nheads_kv        = nheads_kv

        return torch.cat(os, dim=1)   # [total_local_q, nheads, headdim]

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
    ):
        stream = ContextParallelVarlenRingAttention.stream

        (q, k, v, inv_perm, deinterleave_perm, inv_deinterleave_perm,
         cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets) = ctx.saved_tensors

        seqlen_per_rank:  list[int]           = ctx.seqlen_per_rank
        chunk_a_per_rank: list[int]           = ctx.chunk_a_per_rank
        max_local_k:      int                 = ctx.max_local_k
        total_local_k:    int                 = ctx.total_local_k
        heads_stride:     int                 = ctx.heads_stride
        os:               list[torch.Tensor]  = ctx.os
        lses:             list[torch.Tensor]  = ctx.lses
        softmax_scale:    float               = ctx.softmax_scale
        sp_group:         dist.ProcessGroup   = ctx.sp_group
        max_seqlen_q:     int                 = ctx.max_seqlen_q
        max_seqlen_k:     int                 = ctx.max_seqlen_k
        group_size:       int                 = ctx.group_size
        nheads_kv:        int                 = ctx.nheads_kv

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        headdim = k.shape[-1]

        # perm = argsort(inv_perm): maps chunk-major-sample-interleaved → rank-interleaved
        perm = torch.argsort(inv_perm)

        def _make_kv_send(k_h: torch.Tensor, v_h: torch.Tensor) -> torch.Tensor:
            kv = torch.stack([k_h.contiguous(), v_h.contiguous()], dim=0)
            if total_local_k < max_local_k:
                kv = F.pad(kv, (0, 0, 0, 0, 0, max_local_k - total_local_k))
            return kv

        def _unpad_and_cat(bufs: list[torch.Tensor], idx: int) -> torch.Tensor:
            return torch.cat(
                [bufs[r][idx, :seqlen_per_rank[r]] for r in range(sp_size)], dim=0
            )

        # Launch all KV all-gathers upfront on the background stream, one
        # separate buffer per head-stride group so there's no aliasing.
        bwd_per_head_bufs: list[list[torch.Tensor]] = []
        bwd_per_head_events: list[torch.cuda.Event] = []
        with torch.cuda.stream(stream):
            for hi in range(0, nheads_kv, heads_stride):
                buf = [
                    torch.empty(
                        (2, max_local_k, heads_stride, headdim),
                        dtype=k.dtype, device=k.device,
                    )
                    for _ in range(sp_size)
                ]
                dist.all_gather(
                    buf,
                    _make_kv_send(k[:, hi: hi + heads_stride],
                                  v[:, hi: hi + heads_stride]),
                    group=sp_group,
                    async_op=True,
                )
                ev = torch.cuda.Event()
                ev.record(stream)
                bwd_per_head_bufs.append(buf)
                bwd_per_head_events.append(ev)

        dq       = torch.zeros_like(q)
        dk_local = torch.zeros_like(k)
        dv_local = torch.zeros_like(v)

        for head_index in range(0, nheads_kv, heads_stride):
            gi = head_index // heads_stride
            torch.cuda.current_stream().wait_event(bwd_per_head_events[gi])
            kv_buf = bwd_per_head_bufs[gi]

            # Mirror the forward: inv_perm then deinterleave_perm
            gathered_k_half = _unpad_and_cat(kv_buf, 0)[inv_perm][deinterleave_perm]
            gathered_v_half = _unpad_and_cat(kv_buf, 1)[inv_perm][deinterleave_perm]
            gathered_k = torch.cat([gathered_k_half, gathered_k_half], dim=0).contiguous()
            gathered_v = torch.cat([gathered_v_half, gathered_v_half], dim=0).contiguous()

            q_h_start = head_index * group_size
            q_h_end   = (head_index + heads_stride) * group_size
            q_slice   = q[:, q_h_start:q_h_end, :].contiguous()
            do_slice  = do[:, q_h_start:q_h_end, :].contiguous()

            kv_idx = head_index // heads_stride

            dq_h  = torch.empty_like(q_slice)
            dk_gh = torch.empty_like(gathered_k)   # (2*total_global_k, ...)
            dv_gh = torch.empty_like(gathered_v)

            _flash_attn_varlen_backward(
                do_slice, q_slice, gathered_k, gathered_v,
                os[kv_idx], lses[kv_idx],
                dq_h, dk_gh, dv_gh,
                cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets,
                max_seqlen_q, max_seqlen_k,
                softmax_scale,
            )
            dq[:, q_h_start:q_h_end, :] += dq_h

            # Sum gradients from the two copies of K (chunk_a half and chunk_b half)
            total_global_k = gathered_k_half.shape[0]
            dk_gh_sample = dk_gh[:total_global_k] + dk_gh[total_global_k:]
            dv_gh_sample = dv_gh[:total_global_k] + dv_gh[total_global_k:]

            # Reverse deinterleave: sample-major → chunk-major-sample-interleaved
            dk_gh_chunk = dk_gh_sample[inv_deinterleave_perm]
            dv_gh_chunk = dv_gh_sample[inv_deinterleave_perm]

            # Map chunk-major-sample-interleaved → rank-interleaved
            dk_rank_il = dk_gh_chunk[perm]   # [total_global_k, heads_stride, headdim]
            dv_rank_il = dv_gh_chunk[perm]

            # Reduce-scatter with padding: pad each rank's slice to max_local_k
            dk_splits = list(dk_rank_il.to(k.dtype).split(seqlen_per_rank, dim=0))
            dv_splits = list(dv_rank_il.to(v.dtype).split(seqlen_per_rank, dim=0))
            dk_input = torch.stack([
                F.pad(s, (0, 0, 0, 0, 0, max_local_k - s.shape[0]))
                for s in dk_splits
            ])   # (sp_size, max_local_k, heads_stride, headdim)
            dv_input = torch.stack([
                F.pad(s, (0, 0, 0, 0, 0, max_local_k - s.shape[0]))
                for s in dv_splits
            ])

            # Add batch-of-1 dim so gloo's reduce_scatter stub (which cats/splits
            # on dim=1) treats the token axis as dim=1.
            dk_recv4 = torch.empty(
                (1, max_local_k, heads_stride, headdim), dtype=k.dtype, device=k.device
            )
            dv_recv4 = torch.empty_like(dk_recv4)
            dist.reduce_scatter(
                dk_recv4, [s.unsqueeze(0) for s in list(dk_input)], group=sp_group
            )
            dist.reduce_scatter(
                dv_recv4, [s.unsqueeze(0) for s in list(dv_input)], group=sp_group
            )
            dk_recv = dk_recv4.squeeze(0)
            dv_recv = dv_recv4.squeeze(0)

            dk_local[:total_local_k, head_index: head_index + heads_stride, :] += (
                dk_recv[:total_local_k]
            )
            dv_local[:total_local_k, head_index: head_index + heads_stride, :] += (
                dv_recv[:total_local_k]
            )

        torch.cuda.current_stream().wait_stream(stream)

        return (
            dq,
            dk_local,
            dv_local,
            None,  # sp_group
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k_global
            None,  # q_seq_offsets
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # heads_stride
        )


def context_parallel_varlen_ring_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sp_group: dist.ProcessGroup,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_global: torch.Tensor,
    q_seq_offsets: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    heads_stride: int = 1,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Entry point for ring-attention CP with packed varlen Q/K/V.

    query, key, value : [total_local, nheads, headdim]  (already packed varlen)
    cu_seqlens_q      : [2B+1] int32  (2B sub-sequences: B chunk_a then B chunk_b)
    cu_seqlens_k_global : [2B+1] int32
    q_seq_offsets     : [2B] int32
    """
    return (
        ContextParallelVarlenRingAttention.apply(
            query, key, value,
            sp_group,
            cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets,
            max_seqlen_q, max_seqlen_k,
            heads_stride,
        ),
        None,
    )
