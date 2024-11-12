from typing import Optional, Tuple

import torch
import triton
from torch._higher_order_ops.flex_attention import (
    TransformGetItemToIndex,
    create_fw_bw_graph,
    flex_attention,
    flex_attention_autograd,
    flex_attention_backward,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    _apply_kernel_options,
    _convert_block_mask_to_mask,
    _identity,
    _vmap_for_bhqkv,
    create_block_mask,
)

from cornstarch.kernel.triton.any_mask_attn import _attn_any_mask_fwd


def convert_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    assert (
        attention_mask.ndim == 3
    ), f"Expected 4d attention mask of shape (batch_size, q_len, kv_len), got {attention_mask.shape}"

    def custom_mask_mod(b, h, q_idx, kv_idx):
        return attention_mask[b, q_idx, kv_idx].bool()

    block_mask = create_block_mask(
        mask_mod=custom_mask_mod,
        B=attention_mask.shape[0],
        H=num_heads,
        Q_LEN=attention_mask.shape[1],
        KV_LEN=attention_mask.shape[2],
        device=attention_mask.device,
        BLOCK_SIZE=min(attention_mask.shape[1], 128),
        _compile=True,
    )

    return block_mask


def convert_legacy_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    """
    Convert legacy attention mask to BlockMask.
    Legacy attention mask refers to an attention mask with either 0 or 1,
    and later converted to a 4D causal mask.

    Args:
        attention_mask: a 2D tensor of shape (batch_size, seq_len) and type torch.bool or torch.int8.
    """
    assert not getattr(
        attention_mask, "cornstarch_is_bitattention", False
    ), "Expected Non BitAttentionMask"
    assert (
        attention_mask.ndim == 2
    ), f"Expected 2D attention mask, got {attention_mask.ndim}"

    bsz, seq_len = attention_mask.shape

    def causalmask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx >= kv_idx

    return create_block_mask(
        causalmask_mod,
        bsz,
        num_heads,
        seq_len,
        seq_len,
        device=attention_mask.device,
        BLOCK_SIZE=min(seq_len, 128),
        _compile=True,
    )


def convert_bit_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    """
    Convert BitAttentionMask to BlockMask.
    BitAttentionMask refers to an attention mask, where each element is torch.int64 type
    and each bit represents a different types of token to attend.

    Args:
        attention_mask: a 2D tensor of shape (batch_size, seq_len) and type torch.int64.
    """
    assert (
        attention_mask.dtype == torch.int64
    ), f"Expected torch.int64, got {attention_mask.dtype}"
    assert getattr(
        attention_mask, "cornstarch_is_bitattention", False
    ), "Expected BitAttentionMask"
    assert (
        attention_mask.ndim == 2
    ), f"Expected 2D attention mask, got {attention_mask.ndim}"

    bsz, seq_len = attention_mask.shape

    def anymask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        attn_mask = attention_mask.clone()
        is_causal = (attn_mask[b][q_idx] & (1 << 62)) != 0
        attend_non_contiguous = (attn_mask[b][q_idx] & (1 << 61)) != 0

        # Remove 61st and 62nd bits
        attn_mask &= (1 << 61) - 1

        rolled_mask = attn_mask.roll(1, dims=1)
        rolled_mask[:, 0] = 0  # Zero out the first column to avoid boundary issues
        unique_ids = (attn_mask != rolled_mask).cumsum(dim=1)

        return (
            (is_causal & (q_idx >= kv_idx))
            | (
                ~is_causal
                & ~attend_non_contiguous
                & (unique_ids[b][q_idx] == unique_ids[b][kv_idx])
            )
            | (
                ~is_causal
                & attend_non_contiguous
                & (attn_mask[b][q_idx] == attn_mask[b][kv_idx])
            )
        )

    return create_block_mask(
        anymask_mod,
        bsz,
        num_heads,
        seq_len,
        seq_len,
        device=attention_mask.device,
        BLOCK_SIZE=min(seq_len, 128),
        _compile=True,
    )


def _flex_attn_anymask_forward(
    ctx: torch.autograd.function.FunctionCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,  # TODO(@runyu) not used
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
):
    num_heads = q.shape[1]
    if mask.ndim == 2:
        if getattr(mask, "cornstarch_is_bitattention", False):
            block_mask = convert_bit_attention_mask_to_block_mask(
                mask, num_heads
            ).as_tuple()
        else:
            block_mask = convert_legacy_attention_mask_to_block_mask(
                mask, num_heads
            ).as_tuple()
    else:
        block_mask = convert_attention_mask_to_block_mask(mask, num_heads).as_tuple()

    kernel_options = _apply_kernel_options(
        q, k, v, return_lse=True, kernel_options=None
    )
    with TransformGetItemToIndex():
        input_requires_grad = any(t.requires_grad for t in (q, k, v))

        if input_requires_grad and not torch.is_grad_enabled():
            torch.set_grad_enabled(True)

        if torch.is_grad_enabled() and input_requires_grad:
            example_vals = [
                torch.zeros((), dtype=q.dtype, requires_grad=input_requires_grad)
            ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
            fw_graph, bw_graph = create_fw_bw_graph(_identity, example_vals, ())
            assert ctx is not None
        else:
            fw_graph, bw_graph = _identity, None
        assert ctx is not None

        score_mod_other_buffers = ()
        mask_mod_other_buffers = ()

        any_buffer_requires_grad = any(
            buffer.requires_grad
            for buffer in score_mod_other_buffers + mask_mod_other_buffers
        )
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers that require grad are not yet supported."
        ctx._fw_graph = fw_graph
        ctx._joint_graph = bw_graph
        ctx._mask_graph = block_mask[-1]
        # KV_BLOCK_SIZE and Q_BLOCK_SIZE are integers, so can't use ctx.save_for_backward
        # ctx._KV_BLOCK_SIZE = block_mask[8]
        # ctx._Q_BLOCK_SIZE = block_mask[9]
        ctx.scale = softmax_scale
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)
        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = flex_attention(
                q,
                k,
                v,
                fw_graph,
                block_mask,
                softmax_scale,
                kernel_options,
                # score_mod_other_buffers,
                # mask_mod_other_buffers,
            )

    return out, logsumexp * 0.6931471805599453, None, None


def _flex_attn_anymask_naive_backward(
    ctx: torch.autograd.function.FunctionCtx,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    mask_info: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
):
    _, mask = mask_info
    dq, dk, dv = _flash_attn_anymask_backward(
        ctx,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        mask,
        dropout_p,
        softmax_scale,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        deterministic,
        rng_state,
    )
    return dq, dk, dv


def _flex_attn_anymask_backward(
    ctx: torch.autograd.function.FunctionCtx,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    mask_info: Tuple[torch.Tensor],
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with TransformGetItemToIndex():

        softmax_lse = softmax_lse / 0.6931471805599453

        grad_softmax_lse, mask = mask_info

        num_heads = q.shape[1]
        if mask.ndim == 2:
            if getattr(mask, "cornstarch_is_bitattention", False):
                block_mask = convert_bit_attention_mask_to_block_mask(
                    mask, num_heads
                ).as_tuple()
            else:
                block_mask = convert_legacy_attention_mask_to_block_mask(
                    mask, num_heads
                ).as_tuple()
        else:
            block_mask = convert_attention_mask_to_block_mask(
                mask, num_heads
            ).as_tuple()

        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = block_mask[:8]
        KV_BLOCK_SIZE = block_mask[8]
        Q_BLOCK_SIZE = block_mask[9]

        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple()
        mask_mod_other_buffers = tuple()

        # We have asserted that other_buffers do not require grad in the forward
        grad_query, grad_key, grad_value = flex_attention_backward(
            q,
            k,
            v,
            out,
            softmax_lse,
            dout,
            grad_softmax_lse,
            fw_graph,
            joint_graph,
            (
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
                KV_BLOCK_SIZE,
                Q_BLOCK_SIZE,
                mask_graph,
            ),
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )

        if dq is not None:
            dq.copy_(grad_query)
        if dk is not None:
            dk.copy_(grad_key)
        if dv is not None:
            dv.copy_(grad_value)

        return grad_query, grad_key, grad_value


class FlexAttnAnyMask(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        mask,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):

        out, softmax_lse, _, _ = _flex_attn_anymask_forward(
            ctx,
            q,
            k,
            v,
            mask,
            softmax_scale,
            dropout_p,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
            return_softmax,
        )

        ctx.save_for_backward(q, k, v, out, softmax_lse, mask)

        ctx.softmax_scale = softmax_scale
        ctx.dropout_p = dropout_p
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

        return out, softmax_lse, None, None

    @staticmethod
    def backward(ctx, dout, grad_lse, *args):
        q, k, v, out, softmax_lse, mask = ctx.saved_tensors
        mask_info = (grad_lse, mask)
        dq, dk, dv = None, None, None
        # dq, dk, dv = _flex_attn_anymask_naive_backward(ctx, dout, q, k, v, out, softmax_lse, dq, dk, dv, mask_info, ctx.dropout_p, ctx.softmax_scale, ctx.window_size_left, ctx.window_size_right, ctx.softcap, ctx.alibi_slopes, ctx.deterministic, None)
        dq, dk, dv = _flex_attn_anymask_backward(
            ctx,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            mask_info,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            None,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def _flash_attn_anymask_forward(
    ctx: torch.autograd.function.FunctionCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,  # TODO(@runyu) not used
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
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

    grid = lambda args: (
        triton.cdiv(q.shape[2], args["BLOCK_M"]),
        q.shape[0] * q.shape[1],
        1,
    )
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )

    mask_stride_0 = None if not USE_MASK else mask.stride(0)
    mask_stride_1 = None if not USE_MASK else mask.stride(1)
    mask_stride_2 = None if not USE_MASK else mask.stride(2)
    mask_stride_3 = None if not USE_MASK else mask.stride(3)

    # out, softmax_lse, S_dmask, rng_state
    _attn_any_mask_fwd[grid](
        q,
        k,
        v,
        mask,
        softmax_scale,
        M,
        out,  #
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),  #
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),  #
        mask_stride_0,
        mask_stride_1,
        mask_stride_2,
        mask_stride_3,  #
        q.shape[0],
        q.shape[1],  #
        N_CTX=q.shape[2],  #
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,
        USE_MASK=USE_MASK,  #
        **extra_kern_args,
    )

    out, softmax_lse = out, M

    softmax_lse = softmax_lse / 1.44269504  # equal to lse * log2

    # naive implementation:
    # scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    # scores[mask == 0] = -1e4 if scores.dtype == torch.float16 else -1e9
    # softmax_lse = torch.logsumexp(scores.float(), dim=-1)
    # attn_probs = torch.exp(scores.float() - softmax_lse.unsqueeze(-1)).to(scores.dtype)
    # out = torch.matmul(attn_probs, v)

    ctx.save_for_backward(q, k, v, out, softmax_lse, mask)

    # TODO(@runyu) flashattn by tridao will return S_dmask, rng_state, but ringattn only need softmax_lse and out
    return out, softmax_lse, None, None


def _flash_attn_anymask_backward(
    ctx: torch.autograd.function.FunctionCtx,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    mask_info: Tuple[torch.Tensor],
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (grad_output, output, attn_probs, scores, Q_h, K_h, V_h)

    mask = mask_info

    # Step 1: recompute attn_probs
    grad_attn_output = dout  # [batch_size, n_heads, seq_len, d_head]
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    # scores[mask == 0] = float("-inf")
    scores[mask == 0] = -1e4 if scores.dtype == torch.float16 else -1e9
    # attn_probs = torch.softmax(scores, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
    if softmax_lse is not None:
        attn_probs = torch.exp(scores - softmax_lse.unsqueeze(-1)).to(scores.dtype)
    else:
        attn_probs = torch.softmax(
            scores, dim=-1
        )  # [batch_size, n_heads, seq_len, seq_len]

    # Step 2: Gradient w.r.t. V
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_V = torch.matmul(attn_probs.transpose(-2, -1), grad_attn_output)
    if dv is None:
        dv = grad_V
    else:
        dv.copy_(grad_V)

    # Step 3: Gradient w.r.t. attention probabilities
    # [batch_size, n_heads, seq_len, d_head] x [batch_size, n_heads, d_head, seq_len]
    grad_attn_probs = torch.matmul(grad_attn_output, v.transpose(-2, -1))

    # Step 4: Gradient w.r.t. scores (before softmax)
    # Softmax gradient: dL/ds = P * (dL/dP - sum(dL/dP * P))
    # where P is attention probabilities and s is scores
    sum_term = torch.sum(grad_attn_probs * attn_probs, dim=-1, keepdim=True)
    grad_scores = attn_probs * (grad_attn_probs - sum_term)
    grad_scores[mask == 0] = 0.0

    # Step 5: Gradient w.r.t. Q
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_Q = torch.matmul(grad_scores, k) * softmax_scale
    if dq is None:
        dq = grad_Q
    else:
        dq.copy_(grad_Q)

    # Step 6: Gradient w.r.t. K
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), q) * softmax_scale
    if dk is None:
        dk = grad_K
    else:
        dk.copy_(grad_K)

    return dq, dk, dv


# NOTE(runyu): anymask backward flashattn in not correct, so we use the naive manual python implementation now
# def _flash_attn_anymask_backward(
#     dout: torch.Tensor,
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     out: torch.Tensor,
#     softmax_lse: torch.Tensor,
#     dq: Optional[torch.Tensor],
#     dk: Optional[torch.Tensor],
#     dv: Optional[torch.Tensor],
#     mask: torch.Tensor,
#     dropout_p: float = 0.0,
#     softmax_scale: float = 1.0,
#     window_size_left: int = -1,
#     window_size_right: int = -1,
#     softcap: float = 0.0,
#     alibi_slopes: Optional[torch.Tensor] = None,
#     deterministic: bool = False,
#     rng_state: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     assert dout.is_contiguous()
#     assert q.stride() == k.stride() == v.stride() == out.stride() == dout.stride()
#     # TODO(@runyu) stride is not good, check this, maybe the bug
#     assert q.shape == k.shape == v.shape, f"Shape mismatch: q: {q.shape}, k: {k.shape}, v: {v.shape}"
#     assert q.shape == out.shape == dout.shape, f"Shape mismatch: q: {q.shape}, out: {out.shape}, dout: {dout.shape}"
#     HEAD_DIM = q.shape[-1]
#     if dq is None:
#         dq = torch.empty_like(q)
#     if dk is None:
#         dk = torch.empty_like(k)
#     if dv is None:
#         dv = torch.empty_like(v)
#     BATCH, N_HEAD, N_CTX = q.shape[:3]
#     PRE_BLOCK = 128
#     NUM_WARPS, NUM_STAGES = 4, 5
#     BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
#     BLK_SLICE_FACTOR = 2
#     RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
#     softmax_lse = softmax_lse * RCP_LN2
#     arg_k = k
#     arg_k = arg_k * (softmax_scale * RCP_LN2)
#     PRE_BLOCK = 128
#     assert N_CTX % PRE_BLOCK == 0, f"N_CTX: {N_CTX}, PRE_BLOCK: {PRE_BLOCK}"
#     pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
#     delta = torch.empty_like(softmax_lse)
#     _attn_any_mask_bwd_preprocess[pre_grid](
#         out, dout,  #
#         delta,  #
#         BATCH, N_HEAD, N_CTX,  #
#         BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
#     )
#     grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

#     _attn_any_mask_bwd[grid](
#         q, arg_k, v, mask, softmax_scale, dout, dq, dk, dv,  #
#         softmax_lse, delta,  #
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
#         mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),  #
#         N_HEAD, N_CTX,  #
#         BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
#         BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
#         BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
#         HEAD_DIM=HEAD_DIM,
#         USE_MASK=True, #
#         num_warps=NUM_WARPS,  #
#         num_stages=NUM_STAGES,  #
#     )

#     return dq, dk, dv


class FlashAttnAnyMask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask, softmax_scale, dropout_p):
        window_size_left, window_size_right, softcap, alibi_slopes, return_softmax = (
            -1,
            -1,
            0.0,
            None,
            False,
        )
        out, softmax_lse, _, _ = _flash_attn_anymask_forward(
            q,
            k,
            v,
            mask,
            softmax_scale,
            dropout_p,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
            return_softmax,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, mask)
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.dropout_p = dropout_p
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.alibi_slopes = alibi_slopes
        ctx.return_softmax = return_softmax
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, mask = ctx.saved_tensors
        dq, dk, dv = None, None, None
        dropout_p = ctx.dropout_p
        softmax_scale = ctx.softmax_scale
        window_size_left = ctx.window_size_left
        window_size_right = ctx.window_size_right
        softcap = ctx.softcap
        alibi_slopes = ctx.alibi_slopes
        dq, dk, dv = _flash_attn_anymask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            mask,
            dropout_p,
            softmax_scale,
            window_size_left,
            window_size_right,
            softcap,
            alibi_slopes,
        )
        return dq, dk, dv, None, None, None


def _flash_attn_casualmask_forward(
    q: torch.Tensor,  # [B, H, N, D]
    k: torch.Tensor,  # [B, H, N, D]
    v: torch.Tensor,  # [B, H, N, D]
    causal: bool = False,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,  # TODO(@runyu) not used
    # window_size_left: int = -1,
    # window_size_right: int = -1,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert q.ndim == 4, f"q.ndim: {q.ndim}, q.shape: {q.shape}"

    if causal:
        mask = torch.tril(
            torch.ones((q.shape[2], q.shape[2]), device=q.device, dtype=torch.uint8)
        )
        mask = torch.broadcast_to(mask, q.shape[:2] + (q.shape[2], q.shape[2]))
    else:
        mask = torch.ones((q.shape[2], q.shape[2]), device=q.device, dtype=torch.uint8)
        mask = torch.broadcast_to(mask, q.shape[:2] + (q.shape[2], q.shape[2]))

    window_size_left, window_size_right = window_size

    out, softmax_lse, _, _ = _flash_attn_anymask_forward(
        q,
        k,
        v,
        mask,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        alibi_slopes=alibi_slopes,
        return_softmax=return_softmax,
    )

    return out, None, None, None, None, softmax_lse, None, None


def _flash_attn_casualmask_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    assert q.ndim == 4, f"q.ndim: {q.ndim}, q.shape: {q.shape}"

    if causal:
        mask = torch.tril(
            torch.ones((q.shape[2], q.shape[2]), device=q.device, dtype=torch.uint8)
        )
        mask = torch.broadcast_to(mask, q.shape[:2] + (q.shape[2], q.shape[2]))
    else:
        mask = torch.ones((q.shape[2], q.shape[2]), device=q.device, dtype=torch.uint8)
        mask = torch.broadcast_to(mask, q.shape[:2] + (q.shape[2], q.shape[2]))

    window_size_left, window_size_right = window_size

    dq, dk, dv = _flash_attn_anymask_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        mask,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    )

    return dq, dk, dv


"""
TODO(@runyu): code commented below is the original flash attn casual mask forward, it is not used now.
But it is useful and will be used in the future.
"""
# def _flash_attn_casualmask_forward(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     causal: bool,
#     softmax_scale: float = 1.0,
#     dropout_p: float = 0.0, # TODO(@runyu) not used
#     # window_size_left: int = -1,
#     # window_size_right: int = -1,
#     window_size: Tuple[int, int] = (-1, -1),
#     alibi_slopes: Optional[torch.Tensor] = None,
#     return_softmax: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     # shape constraints
#     HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
#     # when v is in float8_e5m2 it is transposed.
#     HEAD_DIM_V = v.shape[-1]
#     assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
#     assert HEAD_DIM_K in {16, 32, 64, 128, 256}, f"HEAD_DIM_K: {HEAD_DIM_K}"
#     o = torch.empty_like(q)
#     stage = 3 if causal else 1
#     extra_kern_args = {}

#     grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
#     M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
#     _attn_fwd[grid](
#         q, k, v, softmax_scale, M, o,  #
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
#         k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
#         v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
#         o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
#         q.shape[0], q.shape[1],  #
#         N_CTX=q.shape[2],  #
#         HEAD_DIM=HEAD_DIM_K,  #
#         STAGE=stage,  #
#         **extra_kern_args)

#     out, softmax_lse = o, M
#     softmax_lse = softmax_lse / 1.44269504 # equal to lse * log2

#     # TODO(@runyu) flashattn by tridao will return S_dmask, rng_state, but ringattn only need softmax_lse and out
#     return out, None, None, None, None, softmax_lse, None, None

# def _flash_attn_casualmask_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic):
#     assert dout.is_contiguous()
#     assert q.stride() == k.stride() == v.stride() == out.stride() == dout.stride(), f"q.stride(): {q.stride()}, k.stride(): {k.stride()}, v.stride(): {v.stride()}, out.stride(): {out.stride()}, dout.stride(): {dout.stride()}"
#     # assert q.shape == k.shape == v.shape, f"Shape mismatch: q: {q.shape}, k: {k.shape}, v: {v.shape}"
#     # assert q.shape == out.shape == dout.shape, f"Shape mismatch: q: {q.shape}, out: {out.shape}, dout: {dout.shape}"

#     HEAD_DIM = k.shape[-1]
#     if dq is None:
#         dq = torch.empty_like(q)
#     if dk is None:
#         dk = torch.empty_like(k)
#     if dv is None:
#         dv = torch.empty_like(v)
#     assert dq.shape == q.shape, f"Shape mismatch: dq: {dq.shape}, q: {q.shape}"
#     assert dk.shape == k.shape, f"Shape mismatch: dk: {dk.shape}, k: {k.shape}"
#     assert dv.shape == v.shape, f"Shape mismatch: dv: {dv.shape}, v: {v.shape}"
#     BATCH, N_HEAD, N_CTX = q.shape[:3]
#     PRE_BLOCK = 128
#     NUM_WARPS, NUM_STAGES = 4, 5
#     BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
#     BLK_SLICE_FACTOR = 2
#     RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
#     arg_k = k
#     softmax_lse = softmax_lse * RCP_LN2
#     arg_k = arg_k * (softmax_scale * RCP_LN2)
#     PRE_BLOCK = 128
#     assert N_CTX % PRE_BLOCK == 0, f"N_CTX: {N_CTX}, PRE_BLOCK: {PRE_BLOCK}"
#     pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
#     delta = torch.empty_like(softmax_lse)
#     _attn_bwd_preprocess[pre_grid](
#         out, dout,  #
#         delta,  #
#         BATCH, N_HEAD, N_CTX,  #
#         BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM  #
#     )
#     grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)

#     _attn_bwd[grid](
#         q, arg_k, v, softmax_scale, dout, dq, dk, dv,  #
#         softmax_lse, delta,  #
#         q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
#         N_HEAD, N_CTX,  #
#         BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
#         BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
#         BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
#         HEAD_DIM=HEAD_DIM,
#         num_warps=NUM_WARPS,  #
#         num_stages=NUM_STAGES,  #
#     )

#     return dq, dk, dv


class FlashAttnCasualMask(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        out, _, _, _, _, softmax_lse, _, _ = _flash_attn_casualmask_forward(
            q, k, v, causal, softmax_scale, dropout_p
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.dropout_p = dropout_p
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = None, None, None
        dq, dk, dv = _flash_attn_casualmask_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.alibi_slopes,
            ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_triton_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    return FlashAttnCasualMask.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
