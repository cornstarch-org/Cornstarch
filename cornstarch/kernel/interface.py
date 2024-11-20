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

from cornstarch.kernel.cache.cache_manager import get_cached_kernels
from cornstarch.kernel.naive_attn import _attn_anymask_backward, _attn_anymask_forward


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
        is_causal = (attention_mask[b][q_idx] & (1 << 62)) != 0
        # Remove it as of now. TODO: Add support for non-contiguous attention
        # attend_non_contiguous = (attention_mask[b][q_idx] & (1 << 61)) != 0

        return (is_causal & (q_idx >= kv_idx)) | (
            ~is_causal & (attention_mask[b, q_idx] == attention_mask[b, kv_idx])
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
            out, logsumexp = flex_attention_hop(
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


def _flex_attn_cached_kernel_forward(
    ctx: torch.autograd.function.FunctionCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: BlockMask,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,  # TODO(@runyu) not used
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
):
    assert isinstance(mask, BlockMask)
    block_mask = mask.as_tuple()
    # Extract block mask components
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

    ctx.fwd_kernel = None
    ctx.bwd_kernel = None

    fwd_kernel, bwd_kernel = get_cached_kernels(
        q.size(0),
        q.size(1),
        q.size(2),
        q.size(3),
        q.dtype,
        softmax_scale,
        mask,
    )

    assert fwd_kernel is not None and bwd_kernel is not None

    ctx.fwd_kernel = fwd_kernel
    ctx.bwd_kernel = bwd_kernel

    # Get the raw CUDA stream instead of torch.cuda.Stream
    from torch._C import _cuda_getCurrentRawStream

    stream0 = _cuda_getCurrentRawStream(0)  # or the device index you're using

    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[:3], device=q.device, dtype=torch.float32)

    meta0 = {
        "ROWS_GUARANTEED_SAFE": False,
        "PRESCALE_QK": False,
        "OUTPUT_LOGSUMEXP": True,
        "FLOAT32_PRECISION": "'ieee'",
        "IS_DIVISIBLE": True,
        "SM_SCALE": softmax_scale,
        "GQA_SHARED_HEADS": 1,
        "HAS_FULL_BLOCKS": True,
        "QK_HEAD_DIM": q.size(3),
        "V_HEAD_DIM": v.size(3),
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "SPARSE_Q_BLOCK_SIZE": q.size(3),
        "SPARSE_KV_BLOCK_SIZE": k.size(3),
    }

    fwd_kernel.run(
        q,
        k,
        v,
        lse,
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        mask,
        out,
        grid=torch._inductor.kernel.flex_attention.flex_attention_grid(
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3),
            meta0,
        ),
        stream=stream0,
    )

    return out, lse * 0.6931471805599453, None, None


def _flex_attn_cached_kernel_backward(
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

    softmax_lse = softmax_lse / 0.6931471805599453
    grad_softmax_lse, mask = mask_info

    assert isinstance(mask, BlockMask)
    block_mask = mask.as_tuple()

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

    # Get the raw CUDA stream instead of torch.cuda.Stream
    from torch._C import _cuda_getCurrentRawStream

    stream0 = _cuda_getCurrentRawStream(0)  # or the device index you're using

    meta0 = {
        "ROWS_GUARANTEED_SAFE": False,
        "PRESCALE_QK": False,
        "OUTPUT_LOGSUMEXP": True,
        "FLOAT32_PRECISION": "'ieee'",
        "IS_DIVISIBLE": True,
        "SM_SCALE": ctx.softmax_scale,
        "GQA_SHARED_HEADS": 1,
        "HAS_FULL_BLOCKS": True,
        "QK_HEAD_DIM": q.size(3),
        "V_HEAD_DIM": v.size(3),
        "BLOCK_M1": 64,  # could be tuned
        "BLOCK_N1": 128,  # could be tuned
        "BLOCK_M2": 128,  # could be tuned
        "BLOCK_N2": 64,  # could be tuned
        "SPARSE_Q_BLOCK_SIZE": q.size(3),
        "SPARSE_KV_BLOCK_SIZE": k.size(3),
    }

    dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)

    # Run backward kernel with corrected stream parameter
    ctx.bwd_kernel.run(
        q,
        k,
        v,
        softmax_lse,
        torch.sum(out * dout, dim=-1),  # NOTE(Runyu) this parameter maybe wrong
        dout,
        dq,
        dv,
        kv_num_blocks,
        kv_indices,
        q_num_blocks,
        q_indices,
        full_kv_num_blocks,
        full_kv_indices,
        full_q_num_blocks,
        full_q_indices,
        mask,
        dk,
        grid=torch._inductor.kernel.flex_attention.flex_attention_backward_grid(
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3),
            k.size(1),
            k.size(2),
            meta0,
        ),
        stream=stream0,  # Pass the raw stream
    )

    return dq, dk, dv


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
