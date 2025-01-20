from __future__ import annotations
import inspect
from functools import cache
from typing import Optional

import torch
import torch.distributed as dist
import numpy as np


@cache
def get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def repeat_attention_mask_heads(
    attention_mask: torch.Tensor, num_heads: int, head_dim: int = 1
):
    """
    transform attention mask 3d(B, L, L) to 4d(B, H, L, L)
    An example:
    attention_mask = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 1, 1]])
    repeat_attention_mask_heads(attention_mask, num_heads=2, head_dim=1)
    >>> torch.tensor([[[1, 0, 0], [1, 0, 1], [0, 1, 1]], [[1, 0, 0], [1, 0, 1], [0, 1, 1]]])
    """
    return attention_mask.unsqueeze(dim=head_dim).expand(-1, num_heads, -1, -1)


def convert_bit_attention_mask_to_full_mask(
    attention_mask: torch.Tensor, num_heads: int
):
    assert getattr(
        attention_mask, "cornstarch_is_bitattention", False
    ), "attention_mask should be bit attention mask."
    assert attention_mask.dtype == torch.int64, "attention_mask should be int64."
    num_encoders = int(getattr(attention_mask, "cornstarch_num_encoders"))

    bsz, seq_len = attention_mask.shape

    encoder_output_indices: list[list[int]] = []
    for i in range(num_encoders):
        mask_set_bit = (attention_mask & (1 << i)) != 0
        mask_unset_causal_bit = (attention_mask & (1 << 62)) == 0

        indices = (mask_set_bit & mask_unset_causal_bit).nonzero(as_tuple=False)[:, 1]
        encoder_output_indices.append(indices)

    # Create a 2D causal mask
    causal_mask_2d = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=attention_mask.device)
    )

    # Expand the 2D causal mask to a 3D tensor
    full_causal_mask = causal_mask_2d.unsqueeze(0).expand(bsz, seq_len, seq_len)

    # Identify the indices where the 62nd bit is set for each batch
    bit_62_mask = (attention_mask & (1 << 62)) != 0

    # Create a list of indices where the 62nd bit is set for each batch
    bit_62_list = bit_62_mask.nonzero(as_tuple=True)

    # Set rows not in bit_62_list to 0 using advanced indexing
    mask = torch.ones((bsz, seq_len), dtype=torch.bool, device=attention_mask.device)
    mask[bit_62_list] = False
    full_causal_mask[mask] = 0

    for indices in encoder_output_indices:
        full_causal_mask[:, indices[:, None], indices] = True

    return full_causal_mask.unsqueeze(1).expand(bsz, num_heads, seq_len, seq_len)


SUPPORT_RING_ATTN_DISTRIBUTION_MODE = ["uniform", "zigzag", "random"]


class ContextParallelBatchUtils:
    split_batch_cache: Optional[np.ndarray] = None

    @classmethod
    def clear_split_cache(cls: ContextParallelBatchUtils):
        cls.split_batch_cache = None

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
            return ContextParallelBatchUtils._split_batch_uniform(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == "zigzag":
            return ContextParallelBatchUtils._split_batch_zigzag(
                batch, sp_group, seq_dim, is_label
            )
        elif ring_attn_mode == "random":
            return ContextParallelBatchUtils._split_batch_random(
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
        cls: ContextParallelBatchUtils,
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
    def _split_batch_random(
        cls: ContextParallelBatchUtils,
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
