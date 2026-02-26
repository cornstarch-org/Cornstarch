"""
Tests for cornstarch.kernel.attention.FlashAttentionVarlenFunc.

Three test scenarios:
  1. test_flash_attn_varlen_causal      — single sequence, standard causal (offset=0).
  2. test_flash_attn_varlen_multi_seq   — multiple packed sequences, all offsets=0.
  3. test_flash_attn_varlen_ring_attn   — offset-aware causal for ring attention
                                          (non-zero q_seq_offsets, Q shorter than K).

All tests compare forward output AND backward gradients against PyTorch SDPA.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from cornstarch.kernel.attention import flash_attn_varlen_func, flash_attn_func


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qkv(
    total_tokens_q: int,
    total_tokens_k: int,
    nheads: int,
    headdim: int,
    dtype=torch.float16,
    device="cuda",
    requires_grad: bool = True,
):
    q = torch.randn(total_tokens_q, nheads, headdim, dtype=dtype, device=device,
                    requires_grad=requires_grad)
    k = torch.randn(total_tokens_k, nheads, headdim, dtype=dtype, device=device,
                    requires_grad=requires_grad)
    v = torch.randn(total_tokens_k, nheads, headdim, dtype=dtype, device=device,
                    requires_grad=requires_grad)
    return q, k, v


def _sdpa_causal(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    """Compute causal SDPA for a single sequence, inputs (S, H, D) → (S, H, D)."""
    # SDPA expects (B, H, S, D)
    q_ = q.transpose(0, 1).unsqueeze(0).float()
    k_ = k.transpose(0, 1).unsqueeze(0).float()
    v_ = v.transpose(0, 1).unsqueeze(0).float()
    o = F.scaled_dot_product_attention(q_, k_, v_, scale=scale, is_causal=True)
    return o.squeeze(0).transpose(0, 1).to(q.dtype)   # (S, H, D)


def _sdpa_with_mask(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    mask: torch.Tensor, scale: float
) -> torch.Tensor:
    """Compute masked SDPA for a single sequence.
    q  : (Sq, H, D)
    k  : (Sk, H, D)
    mask: (Sq, Sk) bool (True = attend)
    """
    q_ = q.transpose(0, 1).unsqueeze(0).float()
    k_ = k.transpose(0, 1).unsqueeze(0).float()
    v_ = v.transpose(0, 1).unsqueeze(0).float()
    # attn_mask: True = keep
    attn_mask = mask.unsqueeze(0).unsqueeze(0).float()   # (1,1,Sq,Sk)
    attn_mask = attn_mask.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    o = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, scale=scale)
    return o.squeeze(0).transpose(0, 1).to(q.dtype)   # (Sq, H, D)


# ---------------------------------------------------------------------------
# Test 1: Single sequence, standard causal (offset=0)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seqlen", [16, 64, 128])
@pytest.mark.parametrize("nheads", [1, 4])
@pytest.mark.parametrize("headdim", [16, 32, 64])
def test_flash_attn_varlen_causal(seqlen, nheads, headdim):
    """Single sequence, standard causal masking (offset=0)."""
    device = "cuda"
    dtype  = torch.float16
    scale  = headdim ** (-0.5)

    q, k, v = _make_qkv(seqlen, seqlen, nheads, headdim, dtype, device)

    cu_seqlens_q    = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    cu_seqlens_k    = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q_seq_offsets   = torch.tensor([0],         dtype=torch.int32, device=device)

    # Forward
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        seqlen, seqlen,
        softmax_scale=scale,
    )
    ref = _sdpa_causal(q.detach(), k.detach(), v.detach(), scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2,
                                msg="Forward mismatch (single-seq causal)")

    # Backward
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    dq_tri, dk_tri, dv_tri = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q2, k2, v2 = _make_qkv(seqlen, seqlen, nheads, headdim, dtype, device)
    q2.data.copy_(q.data); k2.data.copy_(k.data); v2.data.copy_(v.data)
    ref2 = _sdpa_causal(q2, k2, v2, scale)
    ref2.backward(grad_out)

    torch.testing.assert_close(dq_tri.float(), q2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dQ mismatch (single-seq causal)")
    torch.testing.assert_close(dk_tri.float(), k2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dK mismatch (single-seq causal)")
    torch.testing.assert_close(dv_tri.float(), v2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dV mismatch (single-seq causal)")


# ---------------------------------------------------------------------------
# Test 2: Multiple packed sequences, all offsets=0
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_lengths", [[8, 12, 6], [16, 16], [4, 8, 4, 8]])
@pytest.mark.parametrize("nheads", [2])
@pytest.mark.parametrize("headdim", [32])
def test_flash_attn_varlen_multi_seq(seq_lengths, nheads, headdim):
    """Multiple packed sequences, all offsets=0 (standard independent causal attn)."""
    device = "cuda"
    dtype  = torch.float16
    scale  = headdim ** (-0.5)

    total = sum(seq_lengths)
    q, k, v = _make_qkv(total, total, nheads, headdim, dtype, device)

    cu_seqlens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.tensor(seq_lengths, device=device).cumsum(0).int()
    q_seq_offsets = torch.zeros(len(seq_lengths), dtype=torch.int32, device=device)

    # Forward
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens, cu_seqlens, q_seq_offsets,
        max(seq_lengths), max(seq_lengths),
        softmax_scale=scale,
    )

    # Reference: per-sequence SDPA then concatenate
    ref_parts = []
    offset = 0
    for s in seq_lengths:
        q_s = q[offset: offset + s].detach()
        k_s = k[offset: offset + s].detach()
        v_s = v[offset: offset + s].detach()
        ref_parts.append(_sdpa_causal(q_s, k_s, v_s, scale))
        offset += s
    ref = torch.cat(ref_parts, dim=0)

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2,
                                msg="Forward mismatch (multi-seq packed causal)")

    # Backward
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    dq_tri = q.grad.clone()
    dk_tri = k.grad.clone()
    dv_tri = v.grad.clone()

    q2, k2, v2 = _make_qkv(total, total, nheads, headdim, dtype, device)
    q2.data.copy_(q.data); k2.data.copy_(k.data); v2.data.copy_(v.data)
    ref2_parts = []
    offset = 0
    for s in seq_lengths:
        q_s = q2[offset: offset + s]
        k_s = k2[offset: offset + s]
        v_s = v2[offset: offset + s]
        ref2_parts.append(_sdpa_causal(q_s, k_s, v_s, scale))
        offset += s
    ref2 = torch.cat(ref2_parts, dim=0)
    ref2.backward(grad_out)

    torch.testing.assert_close(dq_tri.float(), q2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dQ mismatch (multi-seq packed causal)")
    torch.testing.assert_close(dk_tri.float(), k2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dK mismatch (multi-seq packed causal)")
    torch.testing.assert_close(dv_tri.float(), v2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg="dV mismatch (multi-seq packed causal)")


# ---------------------------------------------------------------------------
# Test 3: Offset-aware causal (ring_attn simulation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "k_len,q_len,offset",
    [
        (64,  32, 0),    # rank 0: no prior tokens (offset=0, causal within local)
        (64,  32, 32),   # rank 1: 32 tokens from rank 0 came before (all visible)
        (128, 64, 0),    # rank 0, longer
        (128, 64, 64),   # rank 1, longer
    ],
)
@pytest.mark.parametrize("nheads", [2])
@pytest.mark.parametrize("headdim", [32])
def test_flash_attn_varlen_ring_attn(k_len, q_len, offset, nheads, headdim):
    """
    Ring-attention offset-aware causal test.

    Q is a rank-local slice of length q_len.
    K/V span the full sequence of length k_len.
    q_seq_offsets[0] = offset means Q token lq can attend to K tokens lk <= offset + lq.

    Reference: explicit boolean mask built as (lk <= offset + lq) via SDPA.
    """
    device = "cuda"
    dtype  = torch.float16
    scale  = headdim ** (-0.5)

    q, k, v = _make_qkv(q_len, k_len, nheads, headdim, dtype, device)

    cu_seqlens_q  = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    cu_seqlens_k  = torch.tensor([0, k_len], dtype=torch.int32, device=device)
    q_seq_offsets = torch.tensor([offset],   dtype=torch.int32, device=device)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k, q_seq_offsets,
        q_len, k_len,
        softmax_scale=scale,
    )

    # Build explicit mask: k_local <= offset + q_local
    q_idx = torch.arange(q_len, device=device)    # (Sq,)
    k_idx = torch.arange(k_len, device=device)    # (Sk,)
    mask  = k_idx[None, :] <= (offset + q_idx[:, None])   # (Sq, Sk) bool

    ref = _sdpa_with_mask(q.detach(), k.detach(), v.detach(), mask, scale)

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2,
                                msg=f"Forward mismatch (ring_attn offset={offset})")

    # Backward
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    dq_tri = q.grad.clone()
    dk_tri = k.grad.clone()
    dv_tri = v.grad.clone()

    q2, k2, v2 = _make_qkv(q_len, k_len, nheads, headdim, dtype, device)
    q2.data.copy_(q.data); k2.data.copy_(k.data); v2.data.copy_(v.data)
    ref2 = _sdpa_with_mask(q2, k2, v2, mask, scale)
    ref2.backward(grad_out)

    torch.testing.assert_close(dq_tri.float(), q2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg=f"dQ mismatch (ring_attn offset={offset})")
    torch.testing.assert_close(dk_tri.float(), k2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg=f"dK mismatch (ring_attn offset={offset})")
    torch.testing.assert_close(dv_tri.float(), v2.grad.float(), atol=1e-1, rtol=1e-1,
                                msg=f"dV mismatch (ring_attn offset={offset})")


# ---------------------------------------------------------------------------
# Test 4: flash_attn_func (batched non-varlen wrapper)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_attn_func_batched():
    """flash_attn_func should match per-sequence causal SDPA for a simple batch."""
    B, S, H, D = 2, 32, 4, 32
    device = "cuda"
    dtype  = torch.float16
    scale  = D ** (-0.5)

    q = torch.randn(B, S, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, S, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, S, H, D, dtype=dtype, device=device, requires_grad=True)

    out = flash_attn_func(q, k, v, softmax_scale=scale)  # (B, S, H, D)

    # Reference: per-sample SDPA
    ref_parts = []
    for b in range(B):
        ref_parts.append(_sdpa_causal(q[b].detach(), k[b].detach(), v[b].detach(), scale))
    ref = torch.stack(ref_parts, dim=0)  # (B, S, H, D)

    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2,
                                msg="Forward mismatch (flash_attn_func batched)")
