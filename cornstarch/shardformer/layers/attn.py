import torch
from typing import Optional
import torch.distributed as dist
from cornstarch.shardformer.layers.utils import split_batch_uniform
from cornstarch.shardformer.layers.ring_attn import ring_flash_attn_forward, ring_flash_attn_backward, ring_flash_attn_anymask_forward, ring_flash_attn_anymask_backward
from cornstarch.kernel.interface import _flash_attn_anymask_forward

from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from cornstarch.kernel.interface import _flash_attn_casualmask_forward, _flash_attn_casualmask_backward


class RingAttentionBase(torch.autograd.Function):
    """
    Base class for ring attention.
    support fixed length input.
    """
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sp_group,
        causal,
        return_softmax,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=False,
        window_size_left=-1,
        window_size_right=-1,
        alibi_slopes=None,
        kernel_impl: str = "cuda", # flash attn of tri for cuda, triton flash attn for triton
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
            ATTN_IMPL= _flash_attn_forward if kernel_impl == "cuda" else _flash_attn_casualmask_forward,
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
            ATTN_IMPL= _flash_attn_backward if ctx.kernel_impl == "cuda" else _flash_attn_casualmask_backward,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None
    
    @staticmethod
    def prepare_batch(
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        inputs_embeds: torch.Tensor = None,
        position_ids: Optional[torch.Tensor] = None,
        is_label: bool = False,
    ):
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        if sp_size == 1:
            return inputs_embeds, position_ids
        
        if inputs_embeds is not None:
            inputs_embeds = split_batch_uniform(inputs_embeds, sp_group)
        attention_mask = split_batch_uniform(attention_mask, sp_group)

    @staticmethod
    def attention(        
        q,
        k,
        v,
        sp_group,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=False,
        return_softmax=False,
        kernel_impl: str = "cuda", # flash attn of tri for cuda, triton flash attn for triton
        **kwargs,):
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

        q = q.transpose(1, 2).contiguous() if kernel_impl == "cuda" else q.contiguous()
        k = k.transpose(1, 2).contiguous() if kernel_impl == "cuda" else k.contiguous()
        v = v.transpose(1, 2).contiguous() if kernel_impl == "cuda" else v.contiguous()

        out, softmax_lse = RingAttentionBase.apply(
            q,
            k,
            v,
            sp_group,
            True, # causal
            True, # return_softmax
            dropout_p,
            softmax_scale,
            deterministic,
            -1, # window_size_left
            -1, # window_size_right
            None, # alibi_slopes
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
        pass

class RingAttentionAnyMask(RingAttentionBase):
    """
    support any mask.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        mask,
        sp_group,
        return_softmax,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=False,
        window_size_left=-1,
        window_size_right=-1,
        alibi_slopes=None,
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
        q,
        k,
        v,
        sp_group,
        mask,
        dropout_p=0.0,
        softmax_scale=None,
        deterministic=False,
        return_softmax=False,
        **kwargs,):
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
            True, # return_softmax
            dropout_p,
            softmax_scale,
            deterministic,
            -1, # window_size_left
            -1, # window_size_right
            None, # alibi_slopes
        )

        out = out.contiguous()

        return out if not return_softmax else (out, softmax_lse)

class DoubleRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, mask_info, sp_group, sp_size, inner_ring_size):
        pass

        