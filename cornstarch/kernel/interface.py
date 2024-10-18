import torch
from typing import Optional, Tuple

from kernel.triton.attn import _attn_any_mask_fwd, _attn_any_mask_bwd, _attn_any_mask_bwd_preprocess
import triton

def _flash_attn_anymask_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    dropout_p: float, # TODO(@runyu) not used
    softmax_scale: float,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    USE_MASK = mask is not None
    # when v is in float8_e5m2 it is transposed.
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    out = torch.empty_like(q)

    # TODO: verify this means mask is not None
    stage = 3 if mask is not None else 2
    extra_kern_args = {}

    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    mask_stride_0 = (None if not USE_MASK else mask.stride(0))
    mask_stride_1 = (None if not USE_MASK else mask.stride(1))
    mask_stride_2 = (None if not USE_MASK else mask.stride(2))
    mask_stride_3 = (None if not USE_MASK else mask.stride(3))
        
    # out, softmax_lse, S_dmask, rng_state
    _attn_any_mask_fwd[grid](
        q, k, v, mask, softmax_scale, M, out,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),  #
        mask_stride_0, mask_stride_1, mask_stride_2, mask_stride_3,  #
        q.shape[0], q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  
        USE_MASK=USE_MASK, #
        **extra_kern_args)
    
    out, softmax_lse = out, M

    # TODO(@runyu) flashattn by tridao will return S_dmask, rng_state, but ringattn only need softmax_lse and out
    return out, softmax_lse, None, None

def _flash_attn_anymask_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    mask: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert dout.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == out.stride() == dout.stride()
    HEAD_DIM = q.shape[-1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    PRE_BLOCK = 128
    NUM_WARPS, NUM_STAGES = 4, 5
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
    arg_k = k
    arg_k = arg_k * (softmax_scale * RCP_LN2)
    PRE_BLOCK = 128
    assert N_CTX % PRE_BLOCK == 0
    pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
    delta = torch.empty_like(softmax_lse)
    _attn_any_mask_bwd_preprocess[pre_grid](
        out, dout,  #
        delta,  #
        BATCH, N_HEAD, N_CTX,  #
        BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
    )
    grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        
    _attn_any_mask_bwd[grid](
        q, arg_k, v, mask, softmax_scale, dout, dq, dk, dv,  #
        softmax_lse, delta,  #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),  #
        N_HEAD, N_CTX,  #
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
        BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
        HEAD_DIM=HEAD_DIM,
        USE_MASK=True, #
        num_warps=NUM_WARPS,  #
        num_stages=NUM_STAGES,  #
    )
