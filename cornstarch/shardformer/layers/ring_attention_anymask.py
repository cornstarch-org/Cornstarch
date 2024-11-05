from __future__ import annotations

from typing import Optional

import numpy as np
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

SUPPORT_RING_ATTN_DISTRIBUTION_MODE = ["uniform", "zigzag", "random"]


class RingAttentionAnyMask(RingAttentionBase):
    """
    support any mask.
    """

    random_split_batch_cache: Optional[np.ndarray] = None

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
            attn_impl=_flash_attn_anymask_forward,
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
    def split_batch(
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
        ring_attn_mode: str = "uniform",
    ) -> torch.Tensor:
        if batch is None:
            return None

        assert (
            ring_attn_mode in SUPPORT_RING_ATTN_DISTRIBUTION_MODE
        ), f"Ring attention distribution mode {ring_attn_mode} is not in the supported list {SUPPORT_RING_ATTN_DISTRIBUTION_MODE}"

        if ring_attn_mode == "uniform":
            return RingAttentionAnyMask._split_batch_uniform(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == "zigzag":
            return RingAttentionAnyMask._split_batch_zigzag(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == "random":
            return RingAttentionAnyMask._split_batch_random(
                batch, sp_group, seq_dim, is_label
            )

    @staticmethod
    def _split_batch_uniform(
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
    ) -> torch.Tensor:
        """
        split them evenly by seq_dim
        """

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        if sp_size == 1:
            return batch

        seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1
        seq_len = batch.shape[seq_dim]

        assert (
            seq_len % sp_size == 0
        ), f"Sequence length {seq_len} must be divisible by {sp_size}!"
        split_batch = batch.chunk(sp_size, dim=seq_dim)[sp_rank].contiguous()

        return split_batch

    @staticmethod
    def _split_batch_zigzag(
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
    ) -> torch.Tensor:
        """
        split them using zigzag strategy
        """
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        if sp_size == 1:
            return batch

        seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1
        seq_len = batch.shape[seq_dim]

        assert (
            seq_len % (sp_size * 2) == 0
        ), f"Sequence length {seq_len} must be divisible by {sp_size * 2}!"

        num_elements_per_process = seq_len // sp_size

        indices = torch.arange(seq_len, device=batch.device)

        # Split indices into two halves for the zigzag pattern
        first_half = indices[: seq_len // 2]
        second_half = indices[seq_len // 2 :].flip(0)

        # Stack the two halves and interleave them to form the zigzag pattern
        zigzag_indices = torch.stack((first_half, second_half), dim=1).flatten()

        # Select the range of indices for the current process
        start_idx = sp_rank * num_elements_per_process
        end_idx = start_idx + num_elements_per_process
        process_indices = zigzag_indices[start_idx:end_idx]

        # Select along the specified dimension using the generated zigzag indices
        split_tensor = batch.index_select(seq_dim, process_indices).contiguous()

        # Reshape to maintain the expected structure
        return split_tensor.view(
            *batch.shape[:seq_dim],
            num_elements_per_process,  # Number of elements for this process
            *batch.shape[seq_dim + 1 :],
        )

    @classmethod
    def clear_split_random_cache(cls: RingAttentionAnyMask):
        cls.random_split_batch_cache = None

    @classmethod
    def _split_batch_random(
        cls: RingAttentionAnyMask,
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
    ) -> torch.Tensor:
        """
        split tokens randomly. If the number of tokens is large it is magically balanced.

        This uses a hash function to assign tokens to processes. The hash function is
        `hash(token_index + random_offset) % sp_size == sp_rank`, where `hash()` is a simple
        linear hash function.
        To ensure even distribution, a and mod (sp_size) should be coprime, i.e. their GCD is 1.
        """
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        if sp_size == 1:
            return batch

        seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1
        seq_len = batch.shape[seq_dim]

        if cls.random_split_batch_cache is not None:
            assert cls.random_split_batch_cache.shape == (seq_len,), (
                f"Random split cache shape {cls.random_split_batch_cache.shape} "
                f"does not match the sequence length {seq_len}"
            )
            assignments = cls.random_split_batch_cache
        else:

            def generate_coprime_a(p, mod):
                while True:
                    a = np.random.randint(1, p)
                    if np.gcd(a, mod) == 1:
                        return a

            # Hash function parameters
            p = 2**31  # Modulus
            a = generate_coprime_a(p, sp_size)  # multiplier
            b = np.random.randint(0, p)  # increment
            offset = np.random.randint(0, p)

            token_indices = np.arange(seq_len, dtype=np.int64)  # shape: [seq_len]

            # Compute the hash for each index
            hash_values = (
                a * ((token_indices + offset) % p) + b
            ) % p  # shape: [seq_len]

            # Determine assignment based on hash modulo 'mod'
            assignments = (hash_values % sp_size) == sp_rank  # shape: [seq_len]

            # Cache assignments
            cls.random_split_batch_cache = assignments

        # Extract the indices assigned to this process
        assigned_indices = np.flatnonzero(assignments)  # shape: [num_assigned_indices]
        assigned_indices = torch.as_tensor(
            assigned_indices, dtype=torch.long, device=batch.device
        )

        # Create a slice to index of selected tokens in seq_dim
        slices = [slice(None)] * batch.dim()
        slices[seq_dim] = (
            assigned_indices  # replace only the seq_dim with assigned indices
        )

        # Use advanced indexing with slices to select the tokens
        return batch[slices].contiguous()

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
