"""
Copied and modified from Tri Dao's Flash Attention Triton implementation:
https://github.com/Dao-AILab/flash-attention/blob/f1a73d074002226c42ce65a1df170ecff9f022c0/flash_attn/flash_attn_triton.py

Changes:
- Remove causal attention.
- Implement running arbitrary attention with bitfield attention mask (BAM).
- Implement a block execution skip, where the block is skipped if all the bits in the bitfield are 0 (masked).
- Implement a padded batch execution.
"""

"""
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

import math
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def is_block_masked_out(mask):
    return tl.sum(mask) == 0


@triton.jit
def get_submask_from_bitfield_mask(
    q_bitfield_mask: tl.tensor,
    kv_bitfield_mask: tl.tensor,
    offs_m,
    offs_n,
):
    """
    Given full bitfield mask, materialize the submask for the current block.

    q_bitfield_mask is used to check whether the query tokens are text tokens.
    For text tokens, all previous tokens are attended to, even for non-text tokens.
    For non-text tokens, only tokens with the same modality bit are attended to.

    Args:
        - q_bitfield_mask: tl.tensor (seqlen_q)
        - kv_bitfield_mask: tl.tensor (seqlen_kv)
        - offs_m: tl.tensor (seqlen_q,)
        - offs_n: tl.tensor (seqlen_kv,)

    Returns:
        - out_mask_buffer: tl.tensor (BLOCK_M, BLOCK_N)
    """
    causal_mask = offs_m[:, None] >= offs_n[None, :]

    is_text_token = ((q_bitfield_mask & 1) > 0)[:, None]

    q_modality_bits = (q_bitfield_mask & ((1 << 62) - 1))[:, None]
    kv_modality_bits = (kv_bitfield_mask & ((1 << 62) - 1))[None, :]

    return tl.where(
        (
            (
                causal_mask
                & (is_text_token == True)
                & ((q_modality_bits & kv_modality_bits) > 0)
            )
            | (
                (is_text_token == False)
                & (q_modality_bits == kv_modality_bits)
                & (q_bitfield_mask[:, None] > 0)
            )
        ),
        1,
        0,
    )


@triton.jit
def compute_num_block_computation(
    attention_mask: tl.tensor,
    out: tl.tensor,
    stride_maskb,
    stride_outb,
    seqlen: tl.constexpr,
):
    """
    For a single query block (128 queries), check the number of blocks that need to be computed.
    Each element in the output tensor is the number of blocks that need computation.

    Args:
        - attention_mask: tl.tensor (batch_size, seqlen)
            a bitfield attention mask
        - out: tl.tensor (batch_size, ceil(seqlen / 128))
        - seqlen: tl.constexpr
    """

    BLOCK_SIZE: tl.constexpr = 128

    batch_size = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK_SIZE

    offs_mask = tl.arange(0, BLOCK_SIZE)
    mask_ptr = attention_mask + batch_size * stride_maskb + offset + offs_mask
    out_ptr = out + batch_size * stride_outb + tl.program_id(1)

    q_bitfield_mask = tl.load(mask_ptr, mask=offset + offs_mask < seqlen, other=0)
    num_blocks_to_compute = 0
    for index in range(0, seqlen, BLOCK_SIZE):
        kv_mask_ptr = attention_mask + batch_size * stride_maskb + index + offs_mask
        kv_bitfield_mask = tl.load(
            kv_mask_ptr, mask=index + offs_mask < seqlen, other=0
        )
        block_mask = get_submask_from_bitfield_mask(
            q_bitfield_mask, kv_bitfield_mask, offset + offs_mask, index + offs_mask
        )

        if not is_block_masked_out(block_mask):
            num_blocks_to_compute += 1

    tl.store(out_ptr, num_blocks_to_compute)


def get_num_computation_block_per_query_block(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # attention_mask: (batch_size, seqlen)
    batch_size, seqlen = attention_mask.shape
    num_blocks = math.ceil(seqlen / 128)
    out = torch.empty(
        (batch_size, num_blocks), dtype=torch.int32, device=attention_mask.device
    )

    grid = lambda META: (batch_size, triton.cdiv(seqlen, 128))
    compute_num_block_computation[grid](
        attention_mask, out, attention_mask.stride(0), out.stride(0), seqlen
    )
    return out


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        # This config has a race condition when EVEN_M == False, disabling it for now.
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"]}
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    bitfield_mask,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_maskb,
    stride_offsetb,
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
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    Context_offsets,
    seqlen_qs,
    seqlen_ks,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # [B, H, N, D]
    start_m = tl.program_id(0)  # blockIdx.x
    off_hb = tl.program_id(1)  # blockIdx.y
    off_b = off_hb // nheads  # batch
    off_h = off_hb % nheads  # head
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    seqlen_q = tl.load(seqlen_qs + off_b)
    seqlen_k = tl.load(seqlen_ks + off_b)

    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)

    # [BLOCK_M, BLOCK_HEADDIM]
    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    # offs_m[:, None]->(BLOCK_M, 1); offs_d[None, :]->(1, BLOCK_HEADDIM)
    # offs_m + offs_d -> (BLOCK_M, BLOCK_HEADDIM)

    # [BLOCK_N, BLOCK_HEADDIM]
    k_ptrs = (
        K
        + off_b * stride_kb
        + off_h * stride_kh
        + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    # [BLOCK_N, BLOCK_HEADDIM]
    v_ptrs = (
        V
        + off_b * stride_vb
        + off_h * stride_vh
        + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        # [BLOCK_N,]
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        # [BLOCK_M, BLOCK_N]
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )

    # initialize pointer to m and l
    t_ptrs = (
        TMP + off_hb * seqlen_q_rounded + offs_m
    )  # intermediate results to current batch and head
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_HEADDIM:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )

    context_offsets_ptr = Context_offsets + off_b * stride_offsetb + offs_m
    context_offsets = tl.load(context_offsets_ptr, mask=offs_m < seqlen_q, other=0)
    q_bitfield_mask = tl.load(
        bitfield_mask + off_b * stride_maskb + context_offsets,
        mask=offs_m < seqlen_q,
        other=0,
    )

    # loop over k, v and update accumulator
    # end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k) # if causal, only do end of BLOCK_M, else the whole N
    end_n = seqlen_k
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        kv_bitfield_mask = tl.load(
            bitfield_mask + off_b * stride_maskb + offs_n_curr,
            mask=offs_n_curr < seqlen_k,
            other=0,
        )

        block_mask = get_submask_from_bitfield_mask(
            q_bitfield_mask, kv_bitfield_mask, context_offsets, offs_n_curr
        )

        if not is_block_masked_out(block_mask):
            # -- compute qk ----
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
            # if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

            qk += tl.where(block_mask, 0, float("-inf"))

            if BIAS_TYPE != "none":
                if BIAS_TYPE == "vector":
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(start_n + offs_n) < seqlen_k,
                        other=0.0,
                    ).to(tl.float32)
                    bias = bias[None, :]
                elif BIAS_TYPE == "matrix":
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
                m_ij = tl.maximum(tl.max(qk, 1), lse_i)  # [BLOCK_M]
                p = tl.exp(qk - m_ij[:, None])  # [BLOCK_M, BLOCK_N]
            else:
                m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
                p = tl.exp(qk * softmax_scale - m_ij[:, None])
            l_ij = tl.sum(p, 1)  # [BLOCK_M]

            # scale acc_o
            acc_o_scale = tl.exp(m_i - m_ij)  # [BLOCK_M]

            # # -- update output accumulator --
            # BUG: have to store and immediately load
            tl.store(t_ptrs, acc_o_scale)
            acc_o_scale = tl.load(t_ptrs)  # store and load from TMP for stability
            acc_o = acc_o * acc_o_scale[:, None]  # [BLOCK_M, BLOCK_HEADDIM]
            # update acc_o
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
            acc_o += tl.dot(p, v)  # [BLOCK_M, BLOCK_HEADDIM]

            # -- update statistics
            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)  # [BLOCK_M]
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)  # store and load from TMP for stability
    acc_o = acc_o * o_scale[:, None]  # [BLOCK_M, BLOCK_HEADDIM]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob  # batch
        + off_h * stride_oh  # head
        + (offs_m[:, None] * stride_om + offs_d[None, :])  # [BLOCK_M, BLOCK_HEADDIM]
    )
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
    seqlen_qs,
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

    seqlen_q = tl.load(seqlen_qs + off_b)

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
    delta = tl.sum(o * do, axis=1)  # [BLOCK_M]
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_HEADDIM:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
    else:
        tl.store(
            dv_ptrs,
            dv,
            mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
        )
        tl.store(
            dk_ptrs,
            dk,
            mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
        )


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    off_b,
    Q,
    K,
    V,
    Bias,
    bitfield_mask,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_maskb,
    stride_offsetb,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    Context_offsets,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    # begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    begin_m = 0
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # initialize pointers to value-like data
    q_ptrs = Q + (
        offs_qm[:, None] * stride_qm + offs_d[None, :]
    )  # [BLOCK_M, BLOCK_HEADDIM]
    k_ptrs = K + (
        offs_n[:, None] * stride_kn + offs_d[None, :]
    )  # [BLOCK_N, BLOCK_HEADDIM]
    v_ptrs = V + (
        offs_n[:, None] * stride_vn + offs_d[None, :]
    )  # [BLOCK_N, BLOCK_HEADDIM]
    do_ptrs = DO + (
        offs_qm[:, None] * stride_dom + offs_d[None, :]
    )  # [BLOCK_M, BLOCK_HEADDIM]
    dq_ptrs = DQ + (
        offs_qm[:, None] * stride_dqm + offs_d[None, :]
    )  # [BLOCK_M, BLOCK_HEADDIM]
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)  # shared memory?
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
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

    kv_bitfield_mask = tl.load(
        bitfield_mask + off_b * stride_maskb + offs_n,
        mask=offs_n < seqlen_k,
        other=0,
    )

    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        context_offsets_ptr = Context_offsets + off_b * stride_offsetb + offs_m_curr
        context_offsets = tl.load(context_offsets_ptr)
        q_bitfield_mask = tl.load(
            bitfield_mask + off_b * stride_maskb + context_offsets,
            mask=offs_m_curr < seqlen_q,
            other=0,
        )

        block_mask = get_submask_from_bitfield_mask(
            q_bitfield_mask, kv_bitfield_mask, context_offsets, offs_n
        )

        if not is_block_masked_out(block_mask):
            # load q, k, v, do on-chip
            # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
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
            # if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

            qk = tl.where(block_mask, qk, float("-inf"))

            # if IS_CAUSAL:
            #     qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
            if BIAS_TYPE != "none":
                tl.debug_barrier()  # Race condition otherwise
                if BIAS_TYPE == "vector":
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(
                        tl.float32
                    )
                    bias = bias[None, :]
                elif BIAS_TYPE == "matrix":
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
                qk = qk * softmax_scale + bias
            # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
            # Also wrong for headdim=64.
            # if not (EVEN_M & EVEN_HEADDIM):
            #     tl.debug_barrier()
            lse_i = tl.load(
                LSE + offs_m_curr, mask=(offs_m_curr < seqlen_q), other=0.0
            )  # [BLOCK_M]
            if BIAS_TYPE == "none":
                p = tl.exp(qk * softmax_scale - lse_i[:, None])  # [BLOCK_M, BLOCK_N]
            else:
                p = tl.exp(qk - lse_i[:, None])
            # compute dv
            # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
            # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
            # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
            # the output is correct.

            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            # if EVEN_M:
            #     if EVEN_HEADDIM:
            #         do = tl.load(do_ptrs)
            #     else:
            #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            # else:
            #     if EVEN_HEADDIM:
            #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            #     else:
            #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
            #                                    & (offs_d[None, :] < headdim), other=0.0)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)
            # compute dp = dot(v, do)
            # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
            # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
            # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
            # if not (EVEN_M & EVEN_HEADDIM):
            #     tl.debug_barrier()
            dp = tl.dot(do, tl.trans(v))
            # There's a race condition for headdim=48
            if not EVEN_HEADDIM:
                tl.debug_barrier()
            # compute ds = p * (dp - delta[:, None])
            # Putting the subtraction after the dp matmul (instead of before) is slightly faster
            Di = tl.load(
                D + offs_m_curr, mask=offs_m_curr < seqlen_q, other=0.0
            )  # [BLOCK_M]
            # Converting ds to q.dtype here reduces register pressure and makes it much faster
            # for BLOCK_HEADDIM=128
            # p = softmax(qk), dp = dot(do, vT), Di: log-sum-exp
            # Subtracts the log-sum-exp term from the intermediate gradient dp
            # Element-wise multiplies the softmax scores p with the corrected gradient term
            # ds: the gradient of the scaled scores
            ds = (p * (dp - Di[:, None]) * softmax_scale).to(
                q.dtype
            )  # [BLOCK_M, BLOCK_N]
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds), q)
            # compute dq
            # if not (
            #     EVEN_M & EVEN_HEADDIM
            # ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            #     tl.debug_barrier()
            if not ATOMIC_ADD:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q)
                        & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
            else:  # If we're parallelizing across the seqlen_k dimension
                dq = tl.dot(ds, k)
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
        # end if not is_block_masked_out
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {"EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"]}
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    bitfield_mask,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_maskb,
    stride_offsetb,
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
    nheads,
    Context_offsets,
    seqlen_qs,
    seqlen_ks,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh

    seqlen_q = tl.load(seqlen_qs + off_b)
    seqlen_k = tl.load(seqlen_ks + off_b)

    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                off_b,
                Q,
                K,
                V,
                Bias,
                bitfield_mask,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_maskb,
                stride_offsetb,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                Context_offsets,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            off_b,
            Q,
            K,
            V,
            Bias,
            bitfield_mask,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_maskb,
            stride_offsetb,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            Context_offsets,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _bitfield_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seqlen_qs: torch.Tensor,
    seqlen_ks: torch.Tensor,
    offsets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    mask: torch.Tensor = None,
):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
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

    seqlen_q_rounded = math.ceil(max(seqlen_qs) / 128) * 128
    lse = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    tmp = torch.empty(
        (batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32
    )
    o = torch.full_like(q, torch.nan)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        mask,
        o,
        lse,
        tmp,
        softmax_scale,
        mask.stride(0),
        offsets.stride(0),
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
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        offsets,
        seqlen_qs,
        seqlen_ks,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        BLOCK_HEADDIM,
        # BLOCK_M=BLOCK,
        # BLOCK_N=BLOCK,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _bitfield_attn_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    mask: torch.Tensor = None,
    seqlen_qs: torch.Tensor = None,
    seqlen_ks: torch.Tensor = None,
    offsets: torch.Tensor = None,
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape

    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(max(seqlen_qs) / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)  # Accumulates gradients for q
    delta = torch.empty_like(lse)  # intermediate results for softmax gradient
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    # compute delta for softmax backward: delta = tl.sum(o * do, axis=1)
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
        seqlen_qs,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
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
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)  # broadcast to all heads
    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)
    )

    # BLOCK_M = 128
    # BLOCK_N = 128
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        mask,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        mask.stride(0),
        offsets.stride(0),
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
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        offsets,
        seqlen_qs,
        seqlen_ks,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)


class BitfieldAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        bias=None,
        softmax_scale=None,
        mask=None,
        seqlen_qs: torch.Tensor = None,
        seqlen_ks: torch.Tensor = None,
        offsets: torch.Tensor = None,
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

        batch = q.shape[0]
        if seqlen_qs is None:
            seqlen_q = q.shape[1]
            seqlen_qs = torch.tensor(
                [seqlen_q] * batch, dtype=torch.int64, device=q.device
            )
        if seqlen_ks is None:
            seqlen_k = k.shape[1]
            seqlen_ks = torch.tensor(
                [seqlen_k] * batch, dtype=torch.int64, device=q.device
            )
        if offsets is None:
            offsets = torch.nested.nested_tensor(
                [
                    torch.arange(seqlen_q, dtype=torch.int32, device=q.device)
                    for seqlen_q in seqlen_qs
                ],
                device=q.device,
            ).to_padded_tensor(padding=-1)

        o, lse, ctx.softmax_scale = _bitfield_attn_forward(
            q,
            k,
            v,
            seqlen_qs=seqlen_qs,
            seqlen_ks=seqlen_ks,
            offsets=offsets,
            bias=bias,
            softmax_scale=softmax_scale,
            mask=mask,
        )
        ctx.save_for_backward(
            q, k, v, o, lse, bias, mask, seqlen_qs, seqlen_ks, offsets
        )
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias, mask, seqlen_qs, seqlen_ks, offsets = ctx.saved_tensors
        assert not ctx.needs_input_grad[
            3
        ], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        # with torch.inference_mode():
        dq = torch.full_like(q, torch.nan)
        dk = torch.full_like(k, torch.nan)
        dv = torch.full_like(v, torch.nan)
        _bitfield_attn_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            bias=bias,
            softmax_scale=ctx.softmax_scale,
            mask=mask,
            seqlen_qs=seqlen_qs,
            seqlen_ks=seqlen_ks,
            offsets=offsets,
        )
        return dq, dk, dv, *([None] * 6)


bitfield_attn_func = BitfieldAttnFunc.apply
