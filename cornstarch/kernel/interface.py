from typing import Optional, Tuple

import torch
from torch._higher_order_ops.flex_attention import (
    TransformGetItemToIndex,
    create_fw_bw_graph,
)
from torch._higher_order_ops.flex_attention import (
    flex_attention as flex_attention_hop,
)
from torch._higher_order_ops.flex_attention import (
    flex_attention_backward as flex_attention_backward_hop,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    _apply_kernel_options,
    _identity,
    create_block_mask,
)

from .naive_attn import _attn_anymask_backward, _attn_anymask_forward


# Dynamo is expecting a callable with "__code__" attribute.
# We cannot directly pass hop to it. So we wrap it in a dummy function.
def _flex_attention_hop_wrapper(*args, **kwargs):
    return flex_attention_hop(*args, **kwargs)


flex_attention = torch.compile(_flex_attention_hop_wrapper, fullgraph=True)


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


@torch.cuda.nvtx.range("flex_attn_anymask_forward")
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
        ctx.kernel_options = kernel_options
        ctx._score_mod_other_buffers_len = len(score_mod_other_buffers)

        with torch._C._AutoDispatchBelowAutograd():
            out, logsumexp = flex_attention(
                q,
                k,
                v,
                _identity,
                mask,
                softmax_scale,
                kernel_options,
            )

    return out, logsumexp * 0.6931471805599453, None, None


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

        (
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = mask[:8]
        KV_BLOCK_SIZE = mask[8]
        Q_BLOCK_SIZE = mask[9]

        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = mask[-1]
        scale = softmax_scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple()
        mask_mod_other_buffers = tuple()

        # We have asserted that other_buffers do not require grad in the forward
        grad_query, grad_key, grad_value = flex_attention_backward_hop(
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
        kernel_impl="flexattn",  # triton means flexattn, naive means naive attn impl
    ):

        attn_impl = (
            _flex_attn_anymask_forward
            if kernel_impl == "flexattn"
            else _attn_anymask_forward
        )

        ctx.kernel_impl = kernel_impl

        out, softmax_lse, _, _ = attn_impl(
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
        attn_impl = (
            _flex_attn_anymask_backward
            if ctx.kernel_impl == "flexattn"
            else _attn_anymask_backward
        )
        dq, dk, dv = attn_impl(
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
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None
