from typing import Optional

import torch
import torch.distributed as dist

from cornstarch.kernel.interface import (
    _flash_attn_anymask_forward,
)
from cornstarch.shardformer.layers.ring_attn import (
    ring_flash_attn_anymask_backward,
    ring_flash_attn_anymask_forward,
)

from ._base import RingAttentionBase


class RingAttentionAnyMask(RingAttentionBase):
    """
    support any mask.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        return_softmax: bool,
        dropout_p: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        alibi_slopes: Optional[torch.Tensor] = None,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        sm_scale = softmax_scale
        assert alibi_slopes is None

        out, softmax_lse = ring_flash_attn_anymask_forward(
            sp_group,
            q,
            k,
            v,
            mask,
            softmax_scale=softmax_scale,
            ATTN_IMPL=_flash_attn_anymask_forward,
            dropout_p=dropout_p,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )

        ctx.save_for_backward(q, k, v, out, mask, softmax_lse)
        # ctx.grid = grid
        ctx.group = sp_group
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = q.shape[-1]
        ctx.USE_MASK = True

        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, mask, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_anymask_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            ctx.sm_scale,
            mask,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        mask: torch.Tensor,
        dropout_p: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        return_softmax: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (torch.Tensor): Query tensor. Shape should be [B, nHeads, Sq, D]
            k (torch.Tensor): Key tensor. Shape should be [B, nHeads, Sq, Sq, D]
            v (torch.Tensor): Value tensor. Shape should be [B, nHeads, Sq, Sq, D]
            sp_group (Optional[dist.ProcessGroup]): Process group for sequence parallelism
            dropout_p (float, optional): Dropout probability. Defaults to 0.0.
            softmax_scale (Optional[float], optional): Scaling factor applied prior to softmax.
            deterministic (bool, optional): Whether to force deterministic backward pass. See https://github.com/Dao-AILab/flash-attention/issues/349
            return_softmax (bool, optional): Whether to return the softmax denominator (logsumexp).
            inner_ring_size (Optional[int], optional): Inner ring size of the 2D ring. By default use a heuristic to decide.

        Returns:
            out: Output tensor of shape [B, nHeads, Sq, D] or [T, nHeads, D] if pad_output is False.
            softmax_lse: (if return_softmax is True) Softmax denominator (logsumexp).
                Shape should be [total_q_seqlen, nHeads]
        """

        out, softmax_lse = RingAttentionAnyMask.apply(
            q,
            k,
            v,
            mask,
            sp_group,
            True,  # return_softmax
            dropout_p,
            softmax_scale,
            deterministic,
            -1,  # window_size_left
            -1,  # window_size_right
            None,  # alibi_slopes
        )

        out = out.contiguous()

        return out if not return_softmax else (out, softmax_lse)
