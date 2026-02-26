"""
Triton varlen FlashAttention with offset-aware causal masking.

Supports packed (varlen) sequences via cu_seqlens_q / cu_seqlens_k, and a
per-sequence q_seq_offsets parameter that shifts the causal boundary:

    token at Q local position lq in sequence s can attend to K token at local
    position lk  iff  lk <= q_seq_offsets[s] + lq

Setting q_seq_offsets[s] = 0 recovers standard same-length causal attention.
Setting q_seq_offsets[s] = k_len_s - q_len_s recovers the "bottom-right"
causal convention used by flash_attn_varlen when Q is shorter than K.

Ring-attention rank r uses q_seq_offsets[s] = number of valid K tokens that
come before rank r's portion of sequence s in global order.

Backward algorithm is the standard online-softmax flash-attention backward.

Grid design (avoids cross-sequence programs):
  - Forward : program (p, h) uses start_m_table[p] → flat Q-start within
              exactly one sequence, seqid_table[p] → sequence id.
  - Backward: program (p, h) uses start_n_table[p] / seqid_table_k[p]
              for K blocks.  Q inner-loop stays within the same sequence.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 32


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fwd_varlen_kernel(
    Q, K, V,
    Out, Lse,
    start_m_table,  # [n_programs] int32: flat Q-start for each program
    seqid_table,    # [n_programs] int32: seq id for each program
    cu_seqlens_q,   # [n_seqs + 1] int32
    cu_seqlens_k,   # [n_seqs + 1] int32
    q_seq_offsets,  # [n_seqs]     int32  (causal boundary shift per seq)
    softmax_scale,
    # strides: Q/K/V are [total_tokens, nheads, headdim]
    stride_qm, stride_qh,
    stride_km, stride_kh,
    stride_vm, stride_vh,
    stride_om, stride_oh,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # grid: (n_programs, nheads)
    prog_id = tl.program_id(0)
    off_h   = tl.program_id(1)
    # GQA: map Q head index to K/V head index
    off_h_kv = off_h // (nheads // nheads_kv)

    # Each program is assigned to exactly one sequence via the tables.
    start_m_flat = tl.load(start_m_table + prog_id).to(tl.int32)
    seq_id       = tl.load(seqid_table   + prog_id).to(tl.int32)

    offs_m = start_m_flat + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_seq_start = tl.load(cu_seqlens_q + seq_id).to(tl.int32)
    q_seq_end   = tl.load(cu_seqlens_q + seq_id + 1).to(tl.int32)
    k_seq_start = tl.load(cu_seqlens_k + seq_id).to(tl.int32)
    k_seq_end   = tl.load(cu_seqlens_k + seq_id + 1).to(tl.int32)
    offset      = tl.load(q_seq_offsets + seq_id).to(tl.int32)

    # Local Q positions within this sequence
    lq_offs = offs_m - q_seq_start

    # Load Q block  [BLOCK_M, BLOCK_HEADDIM]
    q_ptrs = Q + (offs_m[:, None] * stride_qm + off_h * stride_qh + offs_d[None, :])
    q_mask = (offs_m[:, None] < q_seq_end) & (offs_m[:, None] >= q_seq_start)
    if BLOCK_HEADDIM == headdim:
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=q_mask & (offs_d[None, :] < headdim), other=0.0)

    # Online softmax state
    lse_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    m_i   = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # K/V iteration over the K sequence for this seq_id
    for start_n in range(k_seq_start, k_seq_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K
        k_ptrs = K + (offs_n[:, None] * stride_km + off_h_kv * stride_kh + offs_d[None, :])
        k_mask = offs_n[:, None] < k_seq_end
        if BLOCK_HEADDIM == headdim:
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=k_mask & (offs_d[None, :] < headdim), other=0.0)

        # QK scores  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * softmax_scale

        # Causal mask: k_local <= offset + q_local, plus bounds checks
        k_local  = (offs_n - k_seq_start).to(tl.int32)
        q_local  = lq_offs.to(tl.int32)
        causal_ok = k_local[None, :] <= (offset + q_local[:, None])
        in_seq_k  = (offs_n[None, :] < k_seq_end) & (offs_n[None, :] >= k_seq_start)
        in_seq_q  = (offs_m[:, None] < q_seq_end) & (offs_m[:, None] >= q_seq_start)
        valid = causal_ok & in_seq_k & in_seq_q
        qk = tl.where(valid, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        # Guard against all-inf rows (fully masked): avoid exp(-inf - (-inf)) = NaN
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p    = tl.exp(qk - m_ij_safe[:, None])
        p    = tl.where(valid, p, 0.0)
        l_ij = tl.sum(p, 1)

        acc_o_scale = tl.exp(m_i - m_ij_safe)
        acc_o = acc_o * acc_o_scale[:, None]

        # Load V
        v_ptrs = V + (offs_n[:, None] * stride_vm + off_h_kv * stride_vh + offs_d[None, :])
        if BLOCK_HEADDIM == headdim:
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        else:
            v = tl.load(v_ptrs, mask=k_mask & (offs_d[None, :] < headdim), other=0.0)
        acc_o += tl.dot(p.to(v.dtype), v)

        # Update running statistics
        l_new = tl.exp(lse_i - m_ij_safe) + l_ij
        lse_i = m_ij_safe + tl.log(tl.where(l_new == 0.0, 1.0, l_new))
        m_i   = m_ij

    # Normalize output
    lse_safe  = tl.where(lse_i == float("-inf"), 0.0, lse_i)
    o_scale   = tl.exp(m_i - lse_safe)
    acc_o     = acc_o * o_scale[:, None]

    # Write LSE  [total_q, nheads]
    lse_ptrs = Lse + offs_m * nheads + off_h
    tl.store(lse_ptrs, lse_i,
             mask=(offs_m < q_seq_end) & (offs_m >= q_seq_start))

    # Write output
    out_ptrs = Out + (offs_m[:, None] * stride_om + off_h * stride_oh + offs_d[None, :])
    out_mask = (offs_m[:, None] < q_seq_end) & (offs_m[:, None] >= q_seq_start)
    if BLOCK_HEADDIM == headdim:
        tl.store(out_ptrs, acc_o.to(Out.dtype.element_ty), mask=out_mask)
    else:
        tl.store(out_ptrs, acc_o.to(Out.dtype.element_ty),
                 mask=out_mask & (offs_d[None, :] < headdim))


# ---------------------------------------------------------------------------
# Backward preprocess: delta = sum_d(O * dO)
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out, DO, Delta,
    stride_om, stride_oh,
    stride_dom, stride_doh,
    start_m_table,  # [n_programs] int32
    seqid_table,    # [n_programs] int32
    cu_seqlens_q,   # [n_seqs+1]   int32
    nheads: int,
    headdim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    prog_id = tl.program_id(0)
    off_h   = tl.program_id(1)

    start_m_flat = tl.load(start_m_table + prog_id).to(tl.int32)
    seq_id       = tl.load(seqid_table   + prog_id).to(tl.int32)
    q_seq_end    = tl.load(cu_seqlens_q + seq_id + 1).to(tl.int32)

    offs_m = start_m_flat + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    mask   = offs_m[:, None] < q_seq_end

    o_ptrs  = Out + offs_m[:, None] * stride_om + off_h * stride_oh + offs_d[None, :]
    do_ptrs = DO  + offs_m[:, None] * stride_dom + off_h * stride_doh + offs_d[None, :]

    if BLOCK_HEADDIM == headdim:
        o  = tl.load(o_ptrs,  mask=mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        dm = mask & (offs_d[None, :] < headdim)
        o  = tl.load(o_ptrs,  mask=dm, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=dm, other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_m * nheads + off_h, delta, mask=offs_m < q_seq_end)


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_varlen_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    Lse, Delta,
    start_n_table,    # [n_k_programs] int32: flat K-start for each program
    seqid_table_k,    # [n_k_programs] int32: seq id for each program
    cu_seqlens_q,
    cu_seqlens_k,
    q_seq_offsets,
    softmax_scale,
    stride_qm, stride_qh,
    stride_km, stride_kh,
    stride_vm, stride_vh,
    stride_dom, stride_doh,
    stride_dqm, stride_dqh,
    stride_dkm, stride_dkh,
    stride_dvm, stride_dvh,
    nheads: int,
    nheads_kv: int,
    headdim: int,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # grid: (n_k_programs, nheads)
    prog_id = tl.program_id(0)
    off_h   = tl.program_id(1)
    # GQA: map Q head index to K/V head index
    off_h_kv = off_h // (nheads // nheads_kv)

    start_n_flat = tl.load(start_n_table  + prog_id).to(tl.int32)
    seq_id       = tl.load(seqid_table_k  + prog_id).to(tl.int32)

    offs_n = start_n_flat + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_seq_start = tl.load(cu_seqlens_q + seq_id).to(tl.int32)
    q_seq_end   = tl.load(cu_seqlens_q + seq_id + 1).to(tl.int32)
    k_seq_start = tl.load(cu_seqlens_k + seq_id).to(tl.int32)
    k_seq_end   = tl.load(cu_seqlens_k + seq_id + 1).to(tl.int32)
    offset      = tl.load(q_seq_offsets + seq_id).to(tl.int32)

    k_mask = offs_n[:, None] < k_seq_end

    # Load K and V for this tile (held throughout)
    k_ptrs = K + (offs_n[:, None] * stride_km + off_h_kv * stride_kh + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vm + off_h_kv * stride_vh + offs_d[None, :])
    if BLOCK_HEADDIM == headdim:
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
    else:
        dm = k_mask & (offs_d[None, :] < headdim)
        k = tl.load(k_ptrs, mask=dm, other=0.0)
        v = tl.load(v_ptrs, mask=dm, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    k_local = (offs_n - k_seq_start).to(tl.int32)

    for start_m in range(q_seq_start, q_seq_end, BLOCK_M):
        offs_m  = start_m + tl.arange(0, BLOCK_M)
        q_local = (offs_m - q_seq_start).to(tl.int32)

        q_ptrs = Q + (offs_m[:, None] * stride_qm + off_h * stride_qh + offs_d[None, :])
        q_mask = offs_m[:, None] < q_seq_end
        if BLOCK_HEADDIM == headdim:
            q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=q_mask & (offs_d[None, :] < headdim), other=0.0)

        # Recompute QK
        qk = tl.dot(q, tl.trans(k)) * softmax_scale

        causal_ok = k_local[None, :] <= (offset + q_local[:, None])
        in_seq_k  = (offs_n[None, :] < k_seq_end) & (offs_n[None, :] >= k_seq_start)
        in_seq_q  = (offs_m[:, None] < q_seq_end) & (offs_m[:, None] >= q_seq_start)
        valid     = causal_ok & in_seq_k & in_seq_q
        qk        = tl.where(valid, qk, float("-inf"))

        # Recompute softmax from stored LSE
        lse_i = tl.load(Lse + offs_m * nheads + off_h,
                        mask=offs_m < q_seq_end, other=0.0)
        lse_safe = tl.where(lse_i == float("-inf"), 0.0, lse_i)
        p = tl.exp(qk - lse_safe[:, None])
        p = tl.where(valid, p, 0.0)

        # dO
        do_ptrs = DO + (offs_m[:, None] * stride_dom + off_h * stride_doh + offs_d[None, :])
        if BLOCK_HEADDIM == headdim:
            do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        else:
            do = tl.load(do_ptrs, mask=q_mask & (offs_d[None, :] < headdim), other=0.0)

        # dV += p^T @ dO
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)

        # dp = dO @ V^T
        dp = tl.dot(do, tl.trans(v))

        # Di
        Di = tl.load(Delta + offs_m * nheads + off_h,
                     mask=offs_m < q_seq_end, other=0.0)

        # ds = p * (dp - Di)
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        ds = tl.where(valid, ds, 0.0)

        # dK += ds^T @ Q
        dk += tl.dot(tl.trans(ds), q)

        # dQ += ds @ K  (atomic add to avoid race between K-block programs)
        dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + off_h * stride_dqh + offs_d[None, :])
        dq_delta = tl.dot(ds, k)
        if BLOCK_HEADDIM == headdim:
            tl.atomic_add(dq_ptrs, dq_delta, mask=q_mask)
        else:
            tl.atomic_add(dq_ptrs, dq_delta,
                          mask=q_mask & (offs_d[None, :] < headdim))

    # Write dK, dV
    dk_ptrs = DK + (offs_n[:, None] * stride_dkm + off_h_kv * stride_dkh + offs_d[None, :])
    dv_ptrs = DV + (offs_n[:, None] * stride_dvm + off_h_kv * stride_dvh + offs_d[None, :])
    if BLOCK_HEADDIM == headdim:
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=k_mask)
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=k_mask)
    else:
        dm = k_mask & (offs_d[None, :] < headdim)
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=dm)
        tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=dm)


# ---------------------------------------------------------------------------
# Helpers: build per-sequence block tables on the Python side
# ---------------------------------------------------------------------------

def _build_block_table(
    cu_seqlens: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (start_table, seqid_table) where each entry corresponds to one
    Triton program, guaranteed to lie entirely within a single sequence.

    start_table[p] = flat token start for program p
    seqid_table[p] = sequence index for program p
    """
    n_seqs = cu_seqlens.shape[0] - 1
    starts, seqids = [], []
    for s in range(n_seqs):
        s_start = int(cu_seqlens[s].item())
        s_end   = int(cu_seqlens[s + 1].item())
        s_len   = s_end - s_start
        n_blocks = max(1, (s_len + block_size - 1) // block_size)
        for b in range(n_blocks):
            starts.append(s_start + b * block_size)
            seqids.append(s)
    return (
        torch.tensor(starts,  dtype=torch.int32, device=device),
        torch.tensor(seqids, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_seq_offsets: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    q, k, v : [total_tokens, nheads, headdim]
    cu_seqlens_q, cu_seqlens_k : [n_seqs+1] int32
    q_seq_offsets : [n_seqs] int32
    Returns: out [total_q, nheads, headdim], lse [total_q, nheads], softmax_scale
    """
    total_q, nheads, headdim = q.shape
    total_k, nheads_kv, _ = k.shape
    assert k.shape == (total_k, nheads_kv, headdim)
    assert v.shape == (total_k, nheads_kv, headdim)
    assert nheads % nheads_kv == 0, f"nheads ({nheads}) must be divisible by nheads_kv ({nheads_kv})"
    assert headdim <= 128
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k.dtype == v.dtype
    assert q.is_cuda

    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)

    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    cu_seqlens_q  = cu_seqlens_q.contiguous()
    cu_seqlens_k  = cu_seqlens_k.contiguous()
    q_seq_offsets = q_seq_offsets.contiguous()

    out = torch.zeros_like(q)
    lse = torch.full((total_q, nheads), float("-inf"), device=q.device, dtype=torch.float32)

    start_m_table, seqid_table = _build_block_table(cu_seqlens_q, BLOCK_M, q.device)
    n_programs = start_m_table.shape[0]

    grid = (n_programs, nheads)
    _fwd_varlen_kernel[grid](
        q, k, v, out, lse,
        start_m_table, seqid_table,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        softmax_scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1),
        nheads, nheads_kv, headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out, lse, softmax_scale


def _flash_attn_varlen_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_seq_offsets: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
):
    total_q, nheads, headdim = q.shape
    total_k, nheads_kv, _ = k.shape
    softmax_scale = softmax_scale or 1.0 / math.sqrt(headdim)
    BLOCK_HEADDIM = max(triton.next_power_of_2(headdim), 16)

    do = do.contiguous()
    assert out.stride(-1) == 1 and do.stride(-1) == 1

    cu_seqlens_q  = cu_seqlens_q.contiguous()
    cu_seqlens_k  = cu_seqlens_k.contiguous()
    q_seq_offsets = q_seq_offsets.contiguous()

    # Build block tables for Q (preprocess) and K (main bwd kernel)
    start_m_table, seqid_table_q = _build_block_table(cu_seqlens_q, BLOCK_M, q.device)
    start_n_table, seqid_table_k = _build_block_table(cu_seqlens_k, BLOCK_N, k.device)
    n_q_programs = start_m_table.shape[0]
    n_k_programs = start_n_table.shape[0]

    delta = torch.empty((total_q, nheads), device=q.device, dtype=torch.float32)
    _bwd_preprocess_do_o_dot[(n_q_programs, nheads)](
        out, do, delta,
        out.stride(0), out.stride(1),
        do.stride(0),  do.stride(1),
        start_m_table, seqid_table_q,
        cu_seqlens_q,
        nheads, headdim,
        BLOCK_M=BLOCK_M,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # tl.atomic_add does not support bf16 / fp16; accumulate dQ in fp32 then
    # cast back.  dk and dv are written with non-atomic stores (one K-block
    # program per tile) so they can stay in the original dtype.
    dq_f32 = torch.zeros(dq.shape, dtype=torch.float32, device=dq.device)
    _bwd_varlen_kernel[(n_k_programs, nheads)](
        q, k, v, do,
        dq_f32, dk, dv,
        lse, delta,
        start_n_table, seqid_table_k,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        softmax_scale,
        q.stride(0),      q.stride(1),
        k.stride(0),      k.stride(1),
        v.stride(0),      v.stride(1),
        do.stride(0),     do.stride(1),
        dq_f32.stride(0), dq_f32.stride(1),
        dk.stride(0),     dk.stride(1),
        dv.stride(0),     dv.stride(1),
        nheads, nheads_kv, headdim,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    dq.copy_(dq_f32.to(dq.dtype))


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class FlashAttentionVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        q_seq_offsets: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float] = None,
    ):
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v)]
        out, lse, scale = _flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
            max_seqlen_q, max_seqlen_k,
            softmax_scale,
        )
        ctx.save_for_backward(q, k, v, out, lse,
                               cu_seqlens_q, cu_seqlens_k, q_seq_offsets)
        ctx.softmax_scale = scale
        ctx.max_seqlen_q  = max_seqlen_q
        ctx.max_seqlen_k  = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, q_seq_offsets = (
            ctx.saved_tensors
        )
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_varlen_backward(
                do, q, k, v, out, lse,
                dq, dk, dv,
                cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
                ctx.max_seqlen_q, ctx.max_seqlen_k,
                ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_seq_offsets: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Varlen FlashAttention with offset-aware causal masking.

    q, k, v         : [total_tokens, nheads, headdim]  (packed varlen)
    cu_seqlens_q    : [n_seqs + 1]  int32
    cu_seqlens_k    : [n_seqs + 1]  int32
    q_seq_offsets   : [n_seqs]      int32  — causal mask: k_local <= q_seq_offsets[i] + q_local
    max_seqlen_q    : int
    max_seqlen_k    : int
    """
    return FlashAttentionVarlenFunc.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        max_seqlen_q, max_seqlen_k,
        softmax_scale,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compatibility wrapper for batched non-varlen inputs.

    q, k, v : [batch, seqlen, nheads, headdim]
    Returns  : [batch, seqlen, nheads, headdim]
    """
    batch, seqlen_q, nheads, headdim = q.shape
    seqlen_k = k.shape[1]
    assert k.shape == (batch, seqlen_k, nheads, headdim)
    assert v.shape == (batch, seqlen_k, nheads, headdim)

    q_flat = q.reshape(batch * seqlen_q, nheads, headdim)
    k_flat = k.reshape(batch * seqlen_k, nheads, headdim)
    v_flat = v.reshape(batch * seqlen_k, nheads, headdim)

    cu_seqlens_q = torch.arange(
        0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device=q.device
    )
    cu_seqlens_k = torch.arange(
        0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device=k.device
    )
    q_seq_offsets = torch.zeros(batch, dtype=torch.int32, device=q.device)

    out_flat = flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        seqlen_q, seqlen_k,
        softmax_scale,
    )
    return out_flat.reshape(batch, seqlen_q, nheads, headdim)
