"""Context-parallel all-gather flash attention.

Each CP rank holds a subsequence of Q, K, V.  Before computing attention,
each rank gathers the full K and V from all CP ranks so that every rank
attends over the complete key-value sequence.  The all-gather is dispatched
per head group into a side CUDA stream to overlap communication with
computation.

On the backward pass, ``dk``/``dv`` gradients are reduce-scattered back so
each rank accumulates only the gradient for its local K/V slice.

Causal support (A')
-------------------
The default path is **non-causal** (full attention), identical to the original
kernel.  When the caller supplies ``causal=True`` together with
``offsets_per_rank`` (the per-rank *global* sequence positions produced by a
:class:`~cornstarch.distributed.context_parallel.splitters.ContextParallelSplitter`),
the kernel reproduces a causal language model under CP without a custom kernel.

The gathered K/V buffer lays the global key columns out in **rank order**
(rank 0's positions, then rank 1's, ...), so it is a permutation of global
order (identity for the uniform splitter).  For each contiguous run ``[a, b)``
of global positions a rank owns (uniform -> one run/rank; zigzag -> two), the
run's queries must attend exactly the global positions ``[0, query_pos]``,
i.e. every key with global position ``< a`` (the *prefix*) plus the run's own
keys ``[a, query_pos]`` (the *diagonal*).  We build the per-run key/value as
``[prefix_keys ++ diagonal_keys]`` (prefix selected from the gathered buffer by
column, diagonal a contiguous slice in ascending global order) and run a single
stock ``_flash_attn_forward(..., causal=True)``.

flash-attn aligns the causal mask to the **bottom-right** when
``seqlen_q < seqlen_k``: query row ``i`` attends key columns
``[0, i + (seqlen_k - seqlen_q)] = [0, i + prefix_len]`` — every prefix key and
the first ``i + 1`` diagonal keys.  Since the diagonal is in ascending global
order, that is exactly the causal set for global position ``a + i``.  Keys with
global position ``>= b`` are simply excluded from the per-run buffer.  This
needs no online-softmax merge (so no extra bf16 rounding) and keeps flash-class
memory (prefix length <= N, no N x N materialization).

Backward mirrors it: one ``_flash_attn_backward(..., causal=True)`` per run
yields the run's ``dq`` and the combined keys' ``dk``/``dv``, which are
scattered into the correct global columns of the full-length ``dgkv`` buffer
(``index_add`` for the scattered prefix, a contiguous slice for the diagonal)
before the existing reduce-scatter.
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
    ``(2, batch, total_seqlen, heads_stride, d)`` tensor laid out in **rank
    order** (rank 0's positions, then rank 1's, ...) and ``events[gi]`` signals
    when that head group's gather is complete.
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


def _contiguous_runs(local_offsets: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """Split a rank's global positions into maximal ``+1``-contiguous runs.

    ``local_offsets`` is the 1-D ``long`` tensor of global sequence positions a
    rank owns (a splitter's ``_offsets_per_rank[rank]``), ascending within each
    run.  Returns ``(q_start, q_len, a, b)`` tuples where ``[q_start,
    q_start+q_len)`` is the run's range in the rank's *local* Q order and
    ``[a, b)`` is the run's *global* position range (``b - a == q_len``).

    Uniform splitting yields one run/rank; zigzag yields two (an early run and a
    late run).
    """
    offs = local_offsets.tolist()
    n = len(offs)
    runs: list[tuple[int, int, int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and offs[j + 1] == offs[j] + 1:
            j += 1
        runs.append((i, j - i + 1, offs[i], offs[j] + 1))
        i = j + 1
    return runs


def _allgather_offsets(
    position_ids: torch.Tensor,
    seqlen_local: int,
    cp_group: dist.ProcessGroup,
    device: torch.device,
) -> list[torch.Tensor]:
    """Reconstruct every rank's global positions by all-gathering ``position_ids``.

    Each rank's local ``position_ids`` are the *global* sequence positions of the
    tokens it owns (the splitter slices ``position_ids`` alongside the inputs, so
    after the split they carry global indices).  The CP split is identical across
    the batch, so the first row suffices.  Lengths may differ per rank, so the
    per-rank counts are gathered first and the positions gathered into matching
    buffers — yielding the same ``offsets_per_rank`` a splitter would produce,
    without threading the splitter instance into the attention callable.
    """
    cp_size = dist.get_world_size(cp_group)
    if position_ids.ndim == 2:
        position_ids = position_ids[0]
    local_pos = position_ids.reshape(-1).to(device=device, dtype=torch.long).contiguous()
    assert local_pos.numel() == seqlen_local, (
        "position_ids length must match the local K/V sequence length."
    )

    len_t = [torch.empty(1, dtype=torch.long, device=device) for _ in range(cp_size)]
    dist.all_gather(
        len_t, torch.tensor(seqlen_local, device=device), group=cp_group
    )
    lens = [int(t.item()) for t in len_t]

    gathered = [
        torch.empty(lens[r], dtype=torch.long, device=device) for r in range(cp_size)
    ]
    dist.all_gather(gathered, local_pos, group=cp_group)
    return gathered


def _run_causal_kv(
    kv_group: torch.Tensor,
    gathered_global_pos: torch.Tensor,
    diag_start: int,
    q_len: int,
    a: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build a run's ``[prefix ++ diagonal]`` K (and V) from the gathered buffer.

    ``kv_group`` is one of ``gathered_kv[gi][0]`` / ``[1]`` shaped
    ``(batch, total_seqlen, heads_stride, d)``.  The diagonal is the contiguous
    slice ``[diag_start, diag_start + q_len)`` (the run's own keys, ascending
    global order); the prefix is every column with global position ``< a``.
    Returns ``(combined, diag_slice, prefix_index)`` where ``prefix_index`` is
    ``None`` when the run starts at global 0.
    """
    diag = kv_group[:, diag_start : diag_start + q_len]
    if a <= 0:
        return diag.contiguous(), diag, None
    prefix_index = (gathered_global_pos < a).nonzero(as_tuple=True)[0]
    if prefix_index.numel() == 0:
        return diag.contiguous(), diag, None
    prefix = kv_group.index_select(1, prefix_index)
    return torch.cat([prefix, diag], dim=1).contiguous(), diag, prefix_index


class ContextParallelFlashAttention(torch.autograd.Function):
    """Custom autograd function that all-gathers K/V before flash attention.

    Q stays local; K and V are gathered across the CP group so every rank
    attends over the full key-value sequence.  The gather is split by head
    groups (``heads_stride`` heads per call) and overlapped with compute via
    a dedicated CUDA stream.  On the backward pass, full-sequence ``dk``/``dv``
    are computed against the gathered K/V and then reduce-scattered back to
    each rank's local slice.

    With ``causal=True`` and ``offsets_per_rank`` the kernel applies a causal
    mask via the per-run prefix+diagonal decomposition described in the module
    docstring; otherwise it runs full (non-causal) attention.
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
        causal: bool = False,
        offsets_per_rank: list[torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run flash attention with all-gathered K/V.

        Tensors are in ``(batch, seq, heads, dim)`` layout.  ``heads_stride``
        controls how many heads are gathered together per collective call.
        ``causal`` selects the causal decomposition (see the class docstring);
        the default is non-causal full attention.  When causal, the per-rank
        global positions come from ``offsets_per_rank`` if given, otherwise they
        are reconstructed by all-gathering ``position_ids`` (each rank's local
        global positions) across the CP group.
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
        rank = dist.get_rank(cp_group)

        if causal:
            if offsets_per_rank is None:
                assert position_ids is not None, (
                    "causal CP attention needs either offsets_per_rank (the "
                    "splitter's per-rank global positions) or position_ids (this "
                    "rank's global positions, all-gathered to recover the rest)."
                )
                offsets_per_rank = _allgather_offsets(
                    position_ids, seqlen_kv, cp_group, k.device
                )
            else:
                offsets_per_rank = [
                    o.to(device=k.device, dtype=torch.long) for o in offsets_per_rank
                ]
            assert len(offsets_per_rank) == cp_size
            seqlen_per_rank = [int(o.numel()) for o in offsets_per_rank]
            assert seqlen_per_rank[rank] == seqlen_kv
            gathered_global_pos = torch.cat(offsets_per_rank)
            seg_start = sum(seqlen_per_rank[:rank])
            runs = _contiguous_runs(offsets_per_rank[rank])
        else:
            seqlen_per_rank_t = [
                torch.empty(1, dtype=torch.long, device=k.device)
                for _ in range(cp_size)
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
            q_g = q[:, :, hi : hi + heads_stride, :].contiguous()

            if not causal:
                o, lse, _, _ = _flash_attn_forward(
                    q_g,
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
                continue

            # Causal: one flash call per contiguous run over [prefix ++ diagonal].
            run_os: list[torch.Tensor] = []
            run_lses: list[torch.Tensor] = []
            for q_start, q_len, a, _b in runs:
                q_run = q_g[:, q_start : q_start + q_len].contiguous()
                diag_start = seg_start + q_start
                k_run, _, _ = _run_causal_kv(
                    gathered_kv[gi][0], gathered_global_pos, diag_start, q_len, a
                )
                v_run, _, _ = _run_causal_kv(
                    gathered_kv[gi][1], gathered_global_pos, diag_start, q_len, a
                )
                o_run, lse_run, _, _ = _flash_attn_forward(
                    q_run,
                    k_run,
                    v_run,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size_left=-1,
                    window_size_right=-1,
                    softcap=0.0,
                    alibi_slopes=None,
                    return_softmax=False,
                )
                run_os.append(o_run)
                run_lses.append(lse_run)

            os.append(torch.cat(run_os, dim=1))
            lses.append(torch.cat(run_lses, dim=2))

        ctx.save_for_backward(q, k, v)
        ctx.seqlen_per_rank = seqlen_per_rank
        ctx.heads_stride = heads_stride
        ctx.os = os
        ctx.lses = lses
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.causal = causal
        if causal:
            ctx.runs = runs
            ctx.seg_start = seg_start
            ctx.gathered_global_pos = gathered_global_pos

        return torch.cat(os, dim=2)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        do: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None, None
    ]:
        stream = ContextParallelFlashAttention._get_stream()

        q, k, v = ctx.saved_tensors
        seqlen_per_rank: list[int] = ctx.seqlen_per_rank
        heads_stride: int = ctx.heads_stride
        os: list[torch.Tensor] = ctx.os
        lses: list[torch.Tensor] = ctx.lses
        softmax_scale: float = ctx.softmax_scale
        cp_group: dist.ProcessGroup = ctx.cp_group
        causal: bool = ctx.causal

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

            dkv = torch.empty((2, batch, seqlen_kv, heads_stride, d), dtype=k.dtype, device=k.device)

            if not causal:
                dq = torch.empty((batch, seqlen_q, heads_stride, d), dtype=q.dtype, device=q.device)
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
            else:
                # Causal: one flash backward per run over its [prefix ++ diagonal]
                # keys.  Scatter each call's dk/dv into the full-length dgkv at the
                # right global columns (index_add for the prefix, contiguous slice
                # for the diagonal).  A column owned by this rank can be both a
                # diagonal key (its own run) and a prefix key (a later run), so the
                # contributions accumulate; fp32 accumulators keep that exact.
                runs: list[tuple[int, int, int, int]] = ctx.runs
                seg_start: int = ctx.seg_start
                gathered_global_pos: torch.Tensor = ctx.gathered_global_pos

                dq = torch.zeros((batch, seqlen_q, heads_stride, d), dtype=torch.float32, device=q.device)
                dgkv_f32 = torch.zeros((2, batch, total_seqlen, heads_stride, d), dtype=torch.float32, device=k.device)

                for q_start, q_len, a, _b in runs:
                    diag_start = seg_start + q_start
                    k_run, _, prefix_index = _run_causal_kv(
                        gathered_kv[gi][0], gathered_global_pos, diag_start, q_len, a
                    )
                    v_run, _, _ = _run_causal_kv(
                        gathered_kv[gi][1], gathered_global_pos, diag_start, q_len, a
                    )
                    seqlen_k_run = k_run.shape[1]
                    p_len = seqlen_k_run - q_len

                    dq_run = torch.empty((batch, q_len, heads_stride, d), dtype=q.dtype, device=q.device)
                    dk_run = torch.empty((batch, seqlen_k_run, heads_stride, d), dtype=k.dtype, device=k.device)
                    dv_run = torch.empty((batch, seqlen_k_run, heads_stride, d), dtype=k.dtype, device=k.device)
                    _flash_attn_backward(
                        dout=do[:, q_start : q_start + q_len, hi : hi + heads_stride, :].contiguous(),
                        q=q[:, q_start : q_start + q_len, hi : hi + heads_stride, :].contiguous(),
                        k=k_run,
                        v=v_run,
                        out=os[gi][:, q_start : q_start + q_len].contiguous(),
                        softmax_lse=lses[gi][:, :, q_start : q_start + q_len].contiguous(),
                        dq=dq_run,
                        dk=dk_run,
                        dv=dv_run,
                        dropout_p=0.0,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size_left=-1,
                        window_size_right=-1,
                        softcap=0.0,
                        alibi_slopes=None,
                        deterministic=False,
                    )

                    dq[:, q_start : q_start + q_len] += dq_run.float()
                    # k_run / v_run are [prefix ++ diagonal]; split the grads back.
                    if p_len > 0:
                        dgkv_f32[0].index_add_(1, prefix_index, dk_run[:, :p_len].float())
                        dgkv_f32[1].index_add_(1, prefix_index, dv_run[:, :p_len].float())
                    dgkv_f32[0][:, diag_start : diag_start + q_len] += dk_run[:, p_len:].float()
                    dgkv_f32[1][:, diag_start : diag_start + q_len] += dv_run[:, p_len:].float()

                dgkv = dgkv_f32.to(k.dtype)
                dq = dq.to(q.dtype)

            # _flash_attn_backward wrote dgkv on the default stream; the side
            # stream must observe those writes before it reduce-scatters them,
            # otherwise the reduce-scatter reads a partially-written dgkv ->
            # corrupted dk/dv.  (The forward has no analogue: _allgather_kv reads
            # the already-materialized inputs k/v, not a default-stream product.)
            stream.wait_stream(torch.cuda.current_stream())
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
                # Record completion of this group's reduce-scatter on the side
                # stream, mirroring the forward's per-head-group event pattern in
                # _allgather_kv.  The default-stream clones below must wait on this
                # event, otherwise dkv (a torch.empty written by the side stream)
                # is cloned before the reduce-scatter copy lands -> reads
                # uninitialized/partial memory (intermittent NaN / corrupted dk/dv).
                evt = torch.cuda.Event()
                evt.record(stream)

            # Block the default stream on this group's reduce-scatter before
            # cloning dkv.  The side stream can still run the next group's
            # collective ahead, preserving the comm/compute overlap.
            torch.cuda.current_stream().wait_event(evt)
            dqs.append(dq.clone())
            dks.append(dkv[0].clone())
            dvs.append(dkv[1].clone())

        # Backstop: ensure all side-stream work is observed before returning.
        torch.cuda.current_stream().wait_stream(stream)

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


def context_parallel_flash_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cp_group: dist.ProcessGroup,
    heads_stride: int = 1,
    causal: bool = False,
    offsets_per_rank: list[torch.Tensor] | None = None,
    position_ids: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Transpose from HF ``(b, h, s, d)`` to flash-attn ``(b, s, h, d)``, run CP attention, transpose back.

    HF passes ``position_ids`` through to the attention interface as a kwarg; in
    causal mode (and without an explicit ``offsets_per_rank``) it is all-gathered
    inside the kernel to recover every rank's global positions.
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = ContextParallelFlashAttention.apply(
        query, key, value, cp_group, heads_stride, causal, offsets_per_rank, position_ids
    )
    return attn_output.transpose(1, 2), None
