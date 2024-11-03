from typing import Optional

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_backward, _flash_attn_forward

from cornstarch.kernel.interface import (
    _flash_attn_casualmask_backward,
    _flash_attn_casualmask_forward,
)
from cornstarch.shardformer.layers.ring_attn import (
    ring_flash_attn_backward,
    ring_flash_attn_forward,
)

from ._base import RingAttentionBase


class RingAttentionFixedlen(RingAttentionBase):
    """
    Base class for ring attention.
    support fixed length input.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        causal: bool,
        return_softmax: bool,
        dropout_p: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        window_size_left: Optional[int] = -1,
        window_size_right: Optional[int] = -1,
        alibi_slopes: Optional[torch.Tensor] = None,
        kernel_impl: Optional[
            str
        ] = "cuda",  # flash attn of tri for cuda, triton flash attn for triton
    ):

        assert kernel_impl in ["cuda", "triton"]

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        assert alibi_slopes is None
        out, softmax_lse = ring_flash_attn_forward(
            sp_group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            ATTN_IMPL=(
                _flash_attn_forward
                if kernel_impl == "cuda"
                else _flash_attn_casualmask_forward
            ),
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = sp_group
        ctx.kernel_impl = kernel_impl
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors

        assert ctx.kernel_impl in ["cuda", "triton"]

        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size_left=ctx.window_size_left,
            window_size_right=ctx.window_size_right,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            ATTN_IMPL=(
                _flash_attn_backward
                if ctx.kernel_impl == "cuda"
                else _flash_attn_casualmask_backward
            ),
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        dropout_p: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        return_softmax: Optional[bool] = False,
        kernel_impl: str = "cuda",  # flash attn of tri for cuda, triton flash attn for triton
        **kwargs,
    ):
        """
        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, nHeads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Sq, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Sq, Sq, D]
            sp_group (dist.ProcessGroup): Process group for sequence parallelism
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            softmax_scale (float, optional): Scaling factor applied prior to softmax.
            deterministic (bool, optional): Whether to force deterministic backward pass. See https://github.com/Dao-AILab/flash-attention/issues/349
            return_softmax (bool, optional): Whether to return the softmax denominator (logsumexp).

        Returns:
            out: Output tensor of shape [B, nHeads, Sq, D] or [T, nHeads, D] if pad_output is False.
            softmax_lse: (if return_softmax is True) Softmax denominator (logsumexp).
                Shape should be [total_q_seqlen, nHeads]
        """

        q = q.transpose(1, 2).contiguous() if kernel_impl == "cuda" else q.contiguous()
        k = k.transpose(1, 2).contiguous() if kernel_impl == "cuda" else k.contiguous()
        v = v.transpose(1, 2).contiguous() if kernel_impl == "cuda" else v.contiguous()

        out, softmax_lse = RingAttentionBase.apply(
            q,
            k,
            v,
            sp_group,
            True,  # causal
            True,  # return_softmax
            dropout_p,
            softmax_scale,
            deterministic,
            -1,  # window_size_left
            -1,  # window_size_right
            None,  # alibi_slopes
            kernel_impl,
        )

        out = out.contiguous()

        return out if not return_softmax else (out, softmax_lse)


class RingAttentionVarlen(RingAttentionBase):
    """
    support variable length input.
    """

    @staticmethod
    def forward(ctx, qkv, mask_info, sp_group, sp_size, inner_ring_size):
        raise NotImplementedError


class DoubleRingAttention(RingAttentionBase):
    @staticmethod
    def forward(ctx, qkv, mask_info, sp_group, sp_size, inner_ring_size):
        raise NotImplementedError
