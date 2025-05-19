"""
Copied from: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py

*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

BLOCK_M = 128
BLOCK_N = 32
NUM_TOKENS_TO_COMPUTE_PER_BLOCK = 4096
NUM_KV_BLOCKS_PER_BLOCK = NUM_TOKENS_TO_COMPUTE_PER_BLOCK // BLOCK_N


@triton.jit
def _materialize_bitfield_mask_block(
    Bitfield_mask,
    off_m,
    off_n,
    seqlen_q,
    seqlen_k,
    indices_q,
    indices_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if indices_q is not None:
        offs_m = tl.load(
            indices_q + off_m + tl.arange(0, BLOCK_M),
            mask=off_m + tl.arange(0, BLOCK_M) < seqlen_q,
            other=seqlen_k,
        )
        offs_n = tl.load(
            indices_k + off_n + tl.arange(0, BLOCK_N),
            mask=off_n + tl.arange(0, BLOCK_N) < seqlen_k,
            other=seqlen_k,
        )
    else:
        offs_m = off_m + tl.arange(0, BLOCK_M)
        offs_n = off_n + tl.arange(0, BLOCK_N)

    q_bitfield_mask = tl.load(Bitfield_mask + offs_m, mask=offs_m < seqlen_k, other=0)
    kv_bitfield_mask = tl.load(Bitfield_mask + offs_n, mask=offs_n < seqlen_k, other=0)

    causal_mask = offs_m[:, None] >= offs_n[None, :]
    is_text_token = ((q_bitfield_mask & 1) > 0)[:, None]

    q_modality_bits = (q_bitfield_mask & ((1 << 62) - 1))[:, None]
    kv_modality_bits = (kv_bitfield_mask & ((1 << 62) - 1))[None, :]

    return (
        causal_mask & is_text_token & ((q_modality_bits & kv_modality_bits) > 0)
    ) | (
        (is_text_token == False)  # noqa: E712
        & (q_modality_bits == kv_modality_bits)
        & (q_modality_bits > 0)
    )


@triton.jit
def _materialize_compressed_mask(
    Mask,
    Out,
    stride_maskb,
    stride_outb,
    stride_outm,
    seqlen_q,
    seqlen_k,
    indices_q,
    indices_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Create a compressed mask from the bitfield mask.
    For the bitfield mask of shape (batch_size, seq_len),
    it creates a compressed mask of shape (batch_size, seq_len // BLOCK_M, seq_len // BLOCK_N),
    where each element is
    - 0 if all bits in the corresponding BLOCK_M x BLOCK_N block are 0 (maksed out)
    - 1 if some bits in the corresponding BLOCK_M x BLOCK_N block are 1 (not masked out)
    - 2 if all bits in the corresponding BLOCK_M x BLOCK_N block are 1
    """
    off_b = tl.program_id(0)
    start_m = tl.program_id(1)
    start_n = tl.program_id(2)

    submask = _materialize_bitfield_mask_block(
        Mask + off_b * stride_maskb,
        start_m * BLOCK_M,
        start_n * BLOCK_N,
        seqlen_q,
        seqlen_k,
        indices_q,
        indices_k,
        BLOCK_M,
        BLOCK_N,
    )

    value_sum = tl.sum(submask.to(tl.int64))

    result = 0
    if value_sum > 0:
        result = 1

    full_mask_value = (
        seqlen_q % BLOCK_M if (start_m + 1) * BLOCK_M > seqlen_q else BLOCK_M
    ) * (seqlen_k % BLOCK_N if (start_n + 1) * BLOCK_N > seqlen_k else BLOCK_N)
    if value_sum == full_mask_value:
        result = 2

    out_ptr = Out + off_b * stride_outb + start_m * stride_outm + start_n
    tl.store(out_ptr, result)


class BitfieldUtils:
    """
    Cache a compressed mask from the bitfield mask.
    compressed_mask_cache: torch.Tensor of shape
        (batch_size, ceil(seqlen_q, BLOCK_M), ceil(seqlen_k, BLOCK_N))
        where each element can be 0, 1, or 2.
        0: all masked out, no need of computation
        1: partially masked out, need to materialize and apply the mask
        2: fully unmasked, no need of materializing and applying the mask
    """

    compressed_mask_cache: Optional[torch.Tensor] = None

    @classmethod
    def clear_cache(cls: BitfieldUtils):
        cls.compressed_mask_cache = None

    @classmethod
    def materialize_compressed_mask_from_bitfield_mask(
        cls: BitfieldUtils, bitfield_mask: torch.Tensor
    ) -> torch.Tensor:
        if cls.compressed_mask_cache is not None:
            return cls.compressed_mask_cache

        batch_size, seq_len = bitfield_mask.shape
        shape = (
            batch_size,
            triton.cdiv(seq_len, BLOCK_M),
            triton.cdiv(seq_len, BLOCK_N),
        )

        out = torch.zeros(
            shape,
            dtype=torch.int8,
            device=bitfield_mask.device,
        )

        _materialize_compressed_mask[shape](
            bitfield_mask,
            out,
            bitfield_mask.stride(0),
            out.stride(0),
            out.stride(1),
            seq_len,
            seq_len,
            None,
            None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        cls.compressed_mask_cache = out

        return out


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N}, num_warps=4, num_stages=1
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Bitfield_mask,  # (batch_size, seqlen)
    Compressed_mask,  # (batch_size, seqlen // BLOCK_M, seqlen // BLOCK_N)
    Partial_out, # (num_kv_blocks, batch, seqlen_q, nheads, d)
    Partial_lse, # (num_kv_blocks, batch, nheads, seqlen_q_rounded)
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bmb,
    stride_cmb,
    stride_cmm,
    stride_pok,
    stride_pob,
    stride_poh,
    stride_pom,
    stride_plsek,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    offsets_q,
    offsets_k,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    NUM_TILES_TO_COMPUTE_PER_BLOCK: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    kv_block_idx = tl.program_id(2)
    off_n = kv_block_idx * BLOCK_N * NUM_TILES_TO_COMPUTE_PER_BLOCK
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    group_size = nheads // nheads_k
    off_hk = off_h // group_size if group_size > 1 else off_h

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_hk * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_hk * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    cm_base_ptr = Compressed_mask + off_b * stride_cmb + start_m * stride_cmm

    # loop over k, v and update accumulator
    off_n = tl.multiple_of(off_n, BLOCK_N)
    end_n = tl.minimum(off_n + BLOCK_N * NUM_TILES_TO_COMPUTE_PER_BLOCK, seqlen_k)
    for start_n in range(off_n, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        compressed_mask = tl.load(cm_base_ptr + start_n // BLOCK_N)
        if compressed_mask > 0:
            # compressed_mask is either 1 or 2. Need computation
            # if the value is 2, no need of masking out

            # -- compute qk ----
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs + start_n * stride_kn)
                else:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=offs_d[None, :] < headdim,
                        other=0.0,
                    )
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    k = tl.load(
                        k_ptrs + start_n * stride_kn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

            if compressed_mask == 1:
                # mask out the qk after materializing the mask block
                mask_block = _materialize_bitfield_mask_block(
                    Bitfield_mask + off_b * stride_bmb,
                    start_m * BLOCK_M,
                    start_n,
                    seqlen_q,
                    seqlen_k,
                    offsets_q,
                    offsets_k,
                    BLOCK_M,
                    BLOCK_N,
                )
                qk += tl.where(mask_block, 0, float("-inf"))

            if BIAS_TYPE != "none":
                if BIAS_TYPE == "vector":
                    if EVEN_N:
                        bias = tl.load(b_ptrs + start_n).to(tl.float32)
                    else:
                        bias = tl.load(
                            b_ptrs + start_n,
                            mask=(start_n + offs_n) < seqlen_k,
                            other=0.0,
                        ).to(tl.float32)
                    bias = bias[None, :]
                elif BIAS_TYPE == "matrix":
                    if EVEN_M & EVEN_N:
                        bias = tl.load(b_ptrs + start_n).to(tl.float32)
                    else:
                        bias = tl.load(
                            b_ptrs + start_n,
                            mask=(offs_m[:, None] < seqlen_q)
                            & ((start_n + offs_n)[None, :] < seqlen_k),
                            other=0.0,
                        ).to(tl.float32)
                # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
                # can then fuse the mult and add into an fma instruction. But if we have bias we need to
                # to multiply with softmax_scale here.
                qk = qk * softmax_scale + bias
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)
                p = tl.exp(qk - m_ij[:, None])
            else:
                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])

            # if all elements are masked for a token, qk output is -inf
            # and softmax(qk) is nan. We need to set the output to 0.0
            p = tl.where(p != p, 0.0, p)
            m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij)

            l_ij = tl.sum(p, 1)

            # scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)

            # # -- update output accumulator --
            acc_o = acc_o * acc_o_scale[:, None]
            # update acc_o
            if (
                EVEN_N & EVEN_M
            ):  # If we just do "if EVEN_N", there seems to be some race condition
                if EVEN_HEADDIM:
                    v = tl.load(v_ptrs + start_n * stride_vn)
                else:
                    v = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=offs_d[None, :] < headdim,
                        other=0.0,
                    )
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=(start_n + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    v = tl.load(
                        v_ptrs + start_n * stride_vn,
                        mask=((start_n + offs_n)[:, None] < seqlen_k)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)

            # -- update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # Scale acc_o to get the final output for this partial block.
    # acc_o is currently (sum_j exp(S_j - m_i) V_j).
    # We want (sum_j exp(S_j - lse_i) V_j) = acc_o * exp(m_i - lse_i).
    # If lse_i is -inf (all probabilities in block are 0), sum_exp_block_norm_by_m_i will be 0.
    # In this case, safe_o_scale becomes 0, and acc_o (which should be 0 if p was all 0s) remains 0.
    # This prevents NaN if m_i and lse_i are both -inf, where exp(m_i - lse_i) would be NaN.
    sum_exp_block_norm_by_m_i = tl.exp(lse_i - m_i)
    safe_o_scale = tl.where(sum_exp_block_norm_by_m_i > 0.0, tl.exp(m_i - lse_i), 0.0)
    acc_o = acc_o * safe_o_scale[:, None]

    # write back partial lse
    lse_ptrs = Partial_lse + kv_block_idx * stride_plsek + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Partial_out
        + kv_block_idx * stride_pok
        + off_b * stride_pob
        + off_h * stride_poh
        + (offs_m[:, None] * stride_pom + offs_d[None, :])
    )

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
            num_warps=4,
            num_stages=1,
        ),
    ],
    key=[
        "CACHE_KEY_SEQLEN_Q",
        "CACHE_KEY_SEQLEN_K",
        "BIAS_TYPE",
        "BLOCK_HEADDIM",
    ],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    Bitfield_mask,
    Compression_mask,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_bmb,
    stride_cmb,
    stride_cmm,
    nheads,
    nheads_k,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    offsets_q,
    offsets_k,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    NUM_TILES_TO_COMPUTE_PER_BLOCK: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hb = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    group_size = nheads // nheads_k
    off_hk = off_h // group_size if group_size > 1 else off_h

    # initialize offsets
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = q_block_idx * BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q
        + q_block_idx * BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK * stride_qm
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_hk * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_hk * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    do_ptrs = (
        DO
        + q_block_idx * BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK * stride_dom
        + off_b * stride_dob
        + off_h * stride_doh
        + (offs_m[:, None] * stride_dom + offs_d[None, :])
    )
    dq_ptrs = (
        DQ
        + q_block_idx * BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK * stride_dqm
        + off_b * stride_dqb
        + off_h * stride_dqh
        + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    )
    dk_ptrs = (
        DK
        + off_b * stride_dkb
        + off_hk * stride_dkh
        + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    )
    dv_ptrs = (
        DV
        + off_b * stride_dvb
        + off_hk * stride_dvh
        + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + q_block_idx * BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK * stride_bm
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded

    Bitfield_mask += off_b * stride_bmb
    Compression_mask += off_b * stride_cmb

    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    cm_base_ptr = Compression_mask + start_n

    # loop over rows
    end_m = tl.minimum(
        begin_m + BLOCK_M * NUM_TILES_TO_COMPUTE_PER_BLOCK,
        tl.cdiv(seqlen_q, BLOCK_M) * BLOCK_M
    )
    for start_m in range(begin_m, end_m, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        compressed_mask = tl.load(cm_base_ptr + start_m // BLOCK_M * stride_cmm)
        if compressed_mask > 0:
            # load q, k, v, do on-chip
            # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
            if EVEN_M & EVEN_HEADDIM:
                q = tl.load(q_ptrs)
            else:
                if EVEN_HEADDIM:
                    q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
                else:
                    q = tl.load(
                        q_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            # recompute p = softmax(qk, dim=-1).T
            qk = tl.dot(q, tl.trans(k))
            # Trying to combine the two masks seem to make the result wrong
            if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
                qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

            if compressed_mask == 1:
                # mask out the qk after materializing the mask block
                mask_block = _materialize_bitfield_mask_block(
                    Bitfield_mask,
                    start_m,
                    start_n * BLOCK_N,
                    seqlen_q,
                    seqlen_k,
                    offsets_q,
                    offsets_k,
                    BLOCK_M,
                    BLOCK_N,
                )
                qk = tl.where(mask_block, qk, float("-inf"))

            if BIAS_TYPE != "none":
                tl.debug_barrier()  # Race condition otherwise
                if BIAS_TYPE == "vector":
                    if EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(
                            tl.float32
                        )
                    bias = bias[None, :]
                elif BIAS_TYPE == "matrix":
                    if EVEN_M & EVEN_N:
                        bias = tl.load(b_ptrs).to(tl.float32)
                    else:
                        bias = tl.load(
                            b_ptrs,
                            mask=(offs_m_curr[:, None] < seqlen_q)
                            & (offs_n[None, :] < seqlen_k),
                            other=0.0,
                        ).to(tl.float32)
                qk = qk * softmax_scale + bias
            # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
            # Also wrong for headdim=64.
            if not (EVEN_M & EVEN_HEADDIM):
                tl.debug_barrier()
            lse_i = tl.load(LSE + offs_m_curr, mask=offs_m_curr < seqlen_q, other=0.0)
            if BIAS_TYPE == "none":
                p = tl.exp(qk * softmax_scale - lse_i[:, None])
            else:
                p = tl.exp(qk - lse_i[:, None])
            # compute dv
            # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
            # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
            # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
            # the output is correct.
            if EVEN_M & EVEN_HEADDIM:
                do = tl.load(do_ptrs)
            else:
                # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
                do = tl.load(
                    do_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q)
                    & (offs_d[None, :] < headdim),
                    other=0.0,
                )

            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            # compute dp = dot(v, do)
            # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
            # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
            # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
            if not (EVEN_M & EVEN_HEADDIM):
                tl.debug_barrier()
            dp = tl.dot(do, tl.trans(v))
            # There's a race condition for headdim=48
            if not EVEN_HEADDIM:
                tl.debug_barrier()
            # compute ds = p * (dp - delta[:, None])
            # Putting the subtraction after the dp matmul (instead of before) is slightly faster
            Di = tl.load(D + offs_m_curr, mask=offs_m_curr < seqlen_q, other=0.0)
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds), q)
            # compute dq
            if not (
                EVEN_M & EVEN_HEADDIM
            ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
                tl.debug_barrier()

            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                    )

        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm

    # write-back
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv)
            tl.atomic_add(dk_ptrs, dk)
        else:
            tl.atomic_add(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.atomic_add(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.atomic_add(
                dv_ptrs,
                dv,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            )
            tl.atomic_add(
                dk_ptrs,
                dk,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
            )


def _bitfield_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bitfield_mask: torch.Tensor,
    compressed_mask: torch.Tensor,
    offsets_q: Optional[torch.Tensor] = None,
    offsets_k: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)"
                " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )

    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    num_kv_blocks = triton.cdiv(seqlen_k, NUM_TOKENS_TO_COMPUTE_PER_BLOCK)
    
    partial_lse = torch.empty(
        (num_kv_blocks, batch, nheads, seqlen_q_rounded),
        device=q.device,
        dtype=torch.float32,
    )
    partial_out = torch.empty(
        (num_kv_blocks, batch, seqlen_q, nheads, d),
        device=q.device,
        dtype=torch.float32,
    )

    final_lse = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    final_out = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = (
        triton.cdiv(seqlen_q, BLOCK_M),
        batch * nheads,
        num_kv_blocks,
    )
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        bitfield_mask,
        compressed_mask,
        partial_out,
        partial_lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        bitfield_mask.stride(0),
        compressed_mask.stride(0),
        compressed_mask.stride(1),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(3),
        partial_out.stride(2),
        partial_lse.stride(0),
        nheads,
        nheads_k,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        offsets_q,
        offsets_k,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        bias_type,
        BLOCK_HEADDIM,
        NUM_TILES_TO_COMPUTE_PER_BLOCK=NUM_KV_BLOCKS_PER_BLOCK
    )

    grid_aggregate = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _aggregate_partial_outputs[grid_aggregate](
        partial_out,
        partial_lse,
        final_out,
        final_lse,
        partial_out.stride(0),  # kv_blocks stride (first dimension)
        partial_out.stride(1),  # batch stride
        partial_out.stride(3),  # head stride
        partial_out.stride(2),  # sequence stride
        partial_lse.stride(0),  # kv_blocks stride (first dimension)
        partial_lse.stride(1),  # batch stride
        partial_lse.stride(2),  # head stride
        final_out.stride(0),
        final_out.stride(2),
        final_out.stride(1),
        final_lse.stride(0),
        final_lse.stride(1),
        num_kv_blocks,
        seqlen_q,
        seqlen_q_rounded,
        d,
        nheads,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    # softmax_scale could have been updated
    return final_out, final_lse, softmax_scale  



def _bitfield_attn_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    bitfield_mask: torch.Tensor,
    compressed_mask: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    offsets_q: Optional[torch.Tensor] = None,
    offsets_k: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    delta = torch.empty_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = (triton.cdiv(seqlen_q, BLOCK_M), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=BLOCK_M,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)"
                " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )

    num_q_blocks = triton.cdiv(seqlen_q, NUM_TOKENS_TO_COMPUTE_PER_BLOCK)
    grid = (triton.cdiv(seqlen_k, BLOCK_N), batch * nheads, num_q_blocks)
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        bitfield_mask,
        compressed_mask,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq.stride(0),
        dq.stride(2),
        dq.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        bitfield_mask.stride(0),
        compressed_mask.stride(0),
        compressed_mask.stride(1),
        nheads,
        nheads_k,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        offsets_q,
        offsets_k,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        bias_type,
        BLOCK_HEADDIM,
        NUM_TILES_TO_COMPUTE_PER_BLOCK=NUM_KV_BLOCKS_PER_BLOCK
    )


class BitfieldAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bitfield_mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
    ):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

        compressed_mask = BitfieldUtils.materialize_compressed_mask_from_bitfield_mask(
            bitfield_mask
        )

        o, lse, ctx.softmax_scale = _bitfield_attn_forward(
            q,
            k,
            v,
            bitfield_mask,
            compressed_mask,
            bias=bias,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(q, k, v, o, lse, bias, bitfield_mask, compressed_mask)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, k, v, o, lse, bias, bitfield_mask, compressed_mask = ctx.saved_tensors
        assert not ctx.needs_input_grad[
            3
        ], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            _bitfield_attn_backward(
                do,
                q,
                k,
                v,
                o,
                bitfield_mask,
                compressed_mask,
                lse,
                dq,
                dk,
                dv,
                bias=bias,
                softmax_scale=ctx.softmax_scale,
            )
        return dq.to(dtype=q.dtype), dk.to(dtype=k.dtype), dv.to(dtype=v.dtype), None, None, None


def bitfield_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bitfield_mask: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    return BitfieldAttentionFunction.apply(q, k, v, bitfield_mask, bias, softmax_scale)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': BLOCK_M}, num_warps=4, num_stages=1),
    ],
    key=['seqlen_q', 'num_kv_blocks'],
)
@triton.heuristics(
    {
        'EVEN_M': lambda args: args['seqlen_q'] % args['BLOCK_M'] == 0,
        'EVEN_HEADDIM': lambda args: args['headdim'] == args['BLOCK_HEADDIM'],
    }
)
@triton.jit
def _aggregate_partial_outputs(
    Partial_out,
    Partial_lse,
    Final_out,
    Final_lse,
    stride_pok,  # kv_blocks stride (first dimension)
    stride_pob,  # batch stride
    stride_poh,  # head stride
    stride_pom,  # sequence stride
    stride_plsek,  # kv_blocks stride (first dimension)
    stride_plseb,  # batch stride
    stride_plseh,  # head stride
    stride_fob,
    stride_foh,
    stride_fom,
    stride_flseb,
    stride_flseh,
    num_kv_blocks,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    nheads,
    EVEN_M: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    
    # Initialize final output accumulator
    acc_final = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    
    # Single pass: use online softmax to accumulate outputs
    for kv_block in range(num_kv_blocks):
        out_ptr = (
            Partial_out
            + kv_block * stride_pok
            + off_b * stride_pob
            + off_h * stride_poh
            + (offs_m[:, None] * stride_pom + offs_d[None, :])
        )
        lse_ptr = (
            Partial_lse
            + kv_block * stride_plsek
            + off_hb * seqlen_q_rounded
            + offs_m
        )
        
        if EVEN_M:
            if EVEN_HEADDIM:
                partial_o = tl.load(out_ptr)
            else:
                partial_o = tl.load(out_ptr, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                partial_o = tl.load(out_ptr, mask=offs_m[:, None] < seqlen_q, other=0.0)
            else:
                partial_o = tl.load(
                    out_ptr,
                    mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        
        partial_lse_val = tl.load(lse_ptr)
        
        # Online softmax update
        # m_ij_orig is the true new maximum: max(running_max, current_block_lse)
        m_ij_orig = tl.maximum(m_i, partial_lse_val)

        # Stabilize m_ij for use in tl.exp to prevent exp(-inf - (-inf)) = nan.
        # If m_ij_orig is -inf, it means all LSEs encountered so far (m_i and partial_lse_val) were -inf.
        # By setting m_ij_stable to 0.0 if m_ij_orig is -inf, we ensure:
        #   - exp(any_lse - m_ij_stable) becomes exp(-inf - 0.0) = 0 if any_lse was -inf.
        #   - This correctly reflects zero probability contributions.
        m_ij_stable = tl.where(m_ij_orig == float("-inf"), 0.0, m_ij_orig)

        # Scale previous accumulator: acc_old * exp(m_old - m_new_stable)
        acc_final_scale = tl.exp(m_i - m_ij_stable)
        acc_final = acc_final * acc_final_scale[:, None]
        
        # Add current block's contribution: partial_o_k * exp(L_k - m_new_stable)
        # partial_o is already (sum_j exp(S_kj - L_k)V_j).
        # If Partial_out was correctly set to 0 for fully masked blocks by _fwd_kernel,
        # and L_k is -inf, then exp_val = exp(-inf - m_ij_stable).
        # If m_ij_stable is 0 (from -inf), exp_val = 0. Then 0 * 0 = 0.
        # If m_ij_stable is > -inf, exp_val is still valid.
        exp_val = tl.exp(partial_lse_val - m_ij_stable)
        acc_final += partial_o * exp_val[:, None]
        
        # Update statistics
        # lse_new = m_new_orig + log( exp(lse_old - m_new_stable) + exp(L_k - m_new_stable) )
        l_i_new_term_prev = tl.exp(lse_i - m_ij_stable) # If lse_i=-inf, m_ij_stable=0 -> 0
        l_i_new = l_i_new_term_prev + exp_val # This is sum of (stabilized) exp terms
        
        lse_i = m_ij_orig + tl.log(l_i_new) # Use m_ij_orig for the LSE definition. If l_i_new is 0, lse_i becomes -inf.
        m_i = m_ij_orig # Update running max with the true maximum for the next iteration
    
    # Rescale final output, avoid division by zero
    # lse_i = m_i + log(sum_k exp(L_k - m_i))
    # sum_exp_normalized effectively is sum_k exp(L_k - m_i)
    sum_exp_normalized = tl.exp(lse_i - m_i)
    
    # If sum_exp_normalized is 0, lse_i is -inf. m_i - lse_i can be nan or inf.
    # Scale should be 0 if sum_exp_normalized is 0 to make acc_final 0.
    safe_scale = tl.where(sum_exp_normalized > 0.0, tl.exp(m_i - lse_i), 0.0)
    acc_final = acc_final * safe_scale[:, None]
    
    # Write back final output
    final_ptr = (
        Final_out
        + off_b * stride_fob
        + off_h * stride_foh
        + offs_m[:, None] * stride_fom
        + offs_d[None, :]
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(final_ptr, acc_final)
        else:
            tl.store(final_ptr, acc_final, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(final_ptr, acc_final, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                final_ptr,
                acc_final,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            )
    
    # Write back final lse
    final_lse_ptr = (
        Final_lse
        + off_hb * seqlen_q_rounded
        + offs_m
    )
    tl.store(final_lse_ptr, lse_i)
