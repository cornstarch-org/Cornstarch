from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


class ContextParallelDistributionMode(Enum):
    UNIFORM = "uniform"
    ZIGZAG = "zigzag"
    MAKESPAN_MIN = "makespan_min"


class ContextParallelBatchSplitUtils:
    split_batch_cache: Optional[np.ndarray] = None

    @classmethod
    def clear_split_cache(cls: ContextParallelBatchSplitUtils):
        cls.split_batch_cache = None

    @staticmethod
    def split_batch(
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
        ring_attn_mode: ContextParallelDistributionMode = ContextParallelDistributionMode.UNIFORM,
    ) -> torch.Tensor:
        if batch is None:
            return None

        if ring_attn_mode == ContextParallelDistributionMode.UNIFORM:
            return ContextParallelBatchSplitUtils._split_batch_uniform(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == ContextParallelDistributionMode.ZIGZAG:
            return ContextParallelBatchSplitUtils._split_batch_zigzag(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == ContextParallelDistributionMode.MAKESPAN_MIN:
            return ContextParallelBatchSplitUtils._split_batch_makespan_minimization(
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

    @classmethod
    def _split_batch_zigzag(
        cls: ContextParallelBatchSplitUtils,
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
        num_elements_per_process = seq_len // sp_size

        assert (
            seq_len % (sp_size * 2) == 0
        ), f"Sequence length {seq_len} must be divisible by {sp_size * 2}!"

        if cls.split_batch_cache is not None:
            assert cls.split_batch_cache.shape == (seq_len,), (
                f"Zigzag split cache shape {cls.split_batch_cache.shape} "
                f"does not match the sequence length {seq_len}"
            )
            assignments = cls.split_batch_cache
        else:
            indices = np.arange(seq_len)

            first_half = indices[: seq_len // 2]
            second_half = indices[seq_len // 2 :][::-1]

            # Stack the two halves and interleave them to form the zigzag pattern
            assignments = np.ravel(np.column_stack((first_half, second_half)))

            # Cache assignments
            cls.split_batch_cache = assignments

        # Select the range of indices for the current process
        start_idx = sp_rank * num_elements_per_process
        end_idx = start_idx + num_elements_per_process
        process_indices = torch.as_tensor(
            assignments[start_idx:end_idx], dtype=torch.long, device=batch.device
        ).detach()

        slices = [slice(None)] * batch.dim()
        slices[seq_dim] = process_indices

        return batch[slices].contiguous()

    @classmethod
    def _split_batch_makespan_minimization(
        cls: ContextParallelBatchSplitUtils,
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
        seq_dim: int = 1,
        is_label: bool = False,
    ) -> torch.Tensor:
        pass

    @classmethod
    def _split_batch_random(
        cls: ContextParallelBatchSplitUtils,
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

        if cls.split_batch_cache is not None:
            assert cls.split_batch_cache.shape == (seq_len,), (
                f"Random split cache shape {cls.split_batch_cache.shape} "
                f"does not match the sequence length {seq_len}"
            )
            assignments = cls.split_batch_cache
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
            cls.split_batch_cache = assignments

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
