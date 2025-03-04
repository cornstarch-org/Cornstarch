from __future__ import annotations

import heapq
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator

from cornstarch.kernel.bitfield_attention import (
    get_num_computation_block_per_query_block,
)
from cornstarch.shardformer.shard.shard_config import ContextParallelDistributionMode


class ContextParallelBatchSplitUtils:
    """
    Context parallel cache for local sequence lengths and offsets per rank.
    Each tensor in the lists is a tensor of sequence lengths or offsets for a rank.
    """

    context_parallel_sequence_lengths_cache: Optional[
        tuple[list[torch.Tensor], list[torch.Tensor]]
    ] = None

    """
    Indices cache to order gatherd key and value tensors in makespan minimization.
    First tensor is used to converted linearly `cat`ed tensor to in-order tensor.
        This guarantees correct computation of attention scores with arbitrary attention mask.
    Second tensor is used to recover the original order of the tensor,
        so that after sharding the tensors can be scattered back to ranks.
    """
    indices_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    @classmethod
    def clear_cache(cls: ContextParallelBatchSplitUtils):
        cls.context_parallel_sequence_lengths_cache = None
        cls.indices_cache = None

    @classmethod
    def set_context_parallel_sequence_lengths_cache(
        cls: ContextParallelBatchSplitUtils,
        sequence_lengths_per_rank: list[list[int]],
        offsets_per_rank: list[np.ndarray] | list[list[np.ndarray]],
    ):
        if isinstance(offsets_per_rank[0], list):
            new_offsets: list[np.ndarray] = []
            for offset in offsets_per_rank:
                new_offsets.append(
                    np.full(
                        (
                            len(offset),
                            max(map(len, offset)),
                        ),
                        -1,
                    )
                )
                for i, off in enumerate(offset):
                    new_offsets[-1][i, : len(off)] = off
            offsets_per_rank = new_offsets

        sequence_lengths_per_rank = [
            torch.tensor(seq_lens, dtype=torch.int64, device="cpu")
            for seq_lens in sequence_lengths_per_rank
        ]

        offsets_per_rank = [
            torch.tensor(offsets, dtype=torch.int32, device="cpu")
            for offsets in offsets_per_rank
        ]

        cls.context_parallel_sequence_lengths_cache = (
            sequence_lengths_per_rank,
            offsets_per_rank,
        )

    @classmethod
    def get_context_parallel_sequence_lengths_cache(
        cls: ContextParallelBatchSplitUtils,
        sp_rank: int = -1,
        device: torch.device = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        if device is None:
            device = get_accelerator().get_current_device()

        assert sp_rank == -1 or sp_rank >= 0
        if cls.context_parallel_sequence_lengths_cache is None:
            return None, None

        seq_lens, offsets = cls.context_parallel_sequence_lengths_cache
        if sp_rank == -1:
            return [s.to(device) for s in seq_lens], [o.to(device) for o in offsets]
        else:
            return seq_lens[sp_rank].to(device), offsets[sp_rank].to(device)

    @classmethod
    def get_permutate_cache(
        cls: ContextParallelBatchSplitUtils,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = get_accelerator().get_current_device()

        if cls.indices_cache is None:
            assert cls.context_parallel_sequence_lengths_cache is not None
            # Generate permuation and inverse permutation tensors
            offsets = torch.cat(cls.context_parallel_sequence_lengths_cache[1], dim=1)
            perm = torch.argsort(
                torch.where(offsets == -1, torch.iinfo(torch.int32).max, offsets),
                dim=1,
            ).to("cpu")

            inverse_perm = torch.empty_like(perm)
            indices_range = (
                torch.arange(perm.shape[1], device="cpu")
                .unsqueeze(0)
                .expand(perm.shape)
            )
            inverse_perm.scatter_(dim=1, index=perm, src=indices_range)

            cls.indices_cache = (perm, inverse_perm)

        perm, inverse_perm = cls.indices_cache
        return perm.to(device), inverse_perm.to(device)

    @staticmethod
    def split_batch(
        batch: torch.Tensor,
        seqlens: torch.Tensor,
        bitfield_attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        dist_mode: ContextParallelDistributionMode,
        is_label: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - torch.Tensor (batch_size, max_seqlen, num_heads, headdim):
                the split batch
                If each split sequence in a batch has different length, the padding is 0.
            - torch.Tensor (batch_size):
                the sequence lengths of the split batch
            - torch.Tensor (batch_size, max_seqlen):
                the indices of tokens in the split batch.
                If each split sequence in a batch has different length, the padding is -1.
        """
        assert seqlens is not None, "Sequence lengths must be provided."

        if dist_mode == ContextParallelDistributionMode.UNIFORM:
            return ContextParallelBatchSplitUtils.split_batch_uniform(
                batch, seqlens, bitfield_attention_mask, sp_group, is_label
            )
        elif dist_mode == ContextParallelDistributionMode.ZIGZAG:
            return ContextParallelBatchSplitUtils.split_batch_zigzag(
                batch, seqlens, bitfield_attention_mask, sp_group, is_label
            )
        elif dist_mode == ContextParallelDistributionMode.MAKESPAN_MIN:
            return ContextParallelBatchSplitUtils.split_batch_makespan_minimization(
                batch, seqlens, bitfield_attention_mask, sp_group, is_label
            )

    @staticmethod
    def split_batch_uniform(
        batch: torch.Tensor,
        seqlens: torch.Tensor,
        bitfield_attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        is_label: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        split them evenly by seq_dim
        """
        assert batch is not None

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        if sp_size == 1:
            return (
                batch,
                seqlens,
                torch.nested.nested_tensor(
                    [
                        torch.arange(seqlen, device=batch.device, dtype=torch.int32)
                        for seqlen in seqlens
                    ]
                ).to_padded_tensor(padding=-1),
            )

        batch_size, seq_len = bitfield_attention_mask.shape
        if seq_len < 128 * sp_size:
            return (
                batch,
                seqlens,
                torch.nested.nested_tensor(
                    [
                        torch.arange(seqlen, device=batch.device, dtype=torch.int32)
                        for seqlen in seqlens
                    ]
                ).to_padded_tensor(padding=-1),
            )

        split_batches = []
        local_seq_lens: list[list[int]] = [[] for _ in range(sp_size)]
        offsets: list[list[np.ndarray]] = [[] for _ in range(sp_size)]

        if (
            ContextParallelBatchSplitUtils.context_parallel_sequence_lengths_cache
            is not None
        ):
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )

            split_batches = [batch[i, offset] for i, offset in enumerate(offsets)]
        else:
            for i in range(batch_size):
                # We chunk batch[i] with dim=0, which equivalent to dim=1 in the original tensor.
                batch_chunks = batch[i, : seqlens[i]].chunk(sp_size, dim=0)
                split_batches.append(batch_chunks[sp_rank])

                chunked_seqlens = [chunk.shape[0] for chunk in batch_chunks]
                for r in range(sp_size):
                    local_seq_lens[r].append(chunked_seqlens[r])

                    start_offset = sum(chunked_seqlens[:r])
                    offsets[r].append(
                        np.arange(
                            start_offset,
                            start_offset + chunked_seqlens[r],
                            dtype=np.int32,
                        )
                    )

            ContextParallelBatchSplitUtils.set_context_parallel_sequence_lengths_cache(
                local_seq_lens, offsets
            )
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )

        return (
            torch.nested.as_nested_tensor(split_batches, device=batch.device)
            .to_padded_tensor(padding=0.0)
            .contiguous(),
            local_seq_lens,
            offsets,
        )

    @staticmethod
    def split_batch_zigzag(
        batch: torch.Tensor,
        seqlens: torch.Tensor,
        bitfield_attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        is_label: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        split them using zigzag strategy
        """
        assert batch is not None

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        if sp_size == 1:
            return (
                batch,
                seqlens,
                torch.nested.nested_tensor(
                    [
                        torch.arange(seqlen, device=batch.device, dtype=torch.int32)
                        for seqlen in seqlens
                    ]
                ).to_padded_tensor(padding=-1),
            )

        batch_size, seq_len = bitfield_attention_mask.shape
        if seq_len < 128 * sp_size:
            return (
                batch,
                seqlens,
                torch.nested.nested_tensor(
                    [
                        torch.arange(seqlen, device=batch.device, dtype=torch.int32)
                        for seqlen in seqlens
                    ]
                ).to_padded_tensor(padding=-1),
            )

        split_batches = []
        local_seq_lens: list[list[int]] = [[] for _ in range(sp_size)]
        offsets: list[list[np.ndarray]] = [[] for _ in range(sp_size)]

        if (
            ContextParallelBatchSplitUtils.context_parallel_sequence_lengths_cache
            is not None
        ):
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )
        else:
            for i in range(batch_size):
                seq_len = seqlens[i].item()
                indices = np.arange(seq_len)

                # Divide indices into 2*sp_size chunks.
                num_chunks = 2 * sp_size
                chunks = np.array_split(indices, num_chunks)

                # Each rank gets two chunks: the chunk at position 'sp_rank' and the symmetric one.
                for r in range(sp_size):
                    first_chunk = chunks[r]
                    second_chunk = chunks[-r - 1]

                    assignments = np.concatenate([first_chunk, second_chunk])

                    local_seq_lens[r].append(len(assignments))
                    offsets[r].append(assignments)

            ContextParallelBatchSplitUtils.set_context_parallel_sequence_lengths_cache(
                local_seq_lens, offsets
            )
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )

        for i in range(batch_size):
            split_batches.append(batch[i, offsets[i]])

        return (
            torch.nested.as_nested_tensor(split_batches, device=batch.device)
            .to_padded_tensor(padding=0.0)
            .contiguous(),
            local_seq_lens,
            offsets,
        )

    @staticmethod
    def split_batch_makespan_minimization(
        batch: torch.Tensor,
        seqlens: torch.Tensor,
        bitfield_attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        is_label: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        split tokens considering their makespan to be minimized.
        Tokens are grouped as blocks to reduce overhead of makespan minimization algorithm.
        """
        assert batch is not None

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        batch_size, seq_len = bitfield_attention_mask.shape
        if sp_size == 1 or seq_len < 128 * sp_size:
            return (
                batch,
                seqlens,
                torch.nested.nested_tensor(
                    [
                        torch.arange(seqlen, device=batch.device, dtype=torch.int32)
                        for seqlen in seqlens
                    ]
                ).to_padded_tensor(padding=-1),
            )

        if (
            ContextParallelBatchSplitUtils.context_parallel_sequence_lengths_cache
            is not None
        ):
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )
        else:
            block_size = 128

            # compress the attention mask, for each 128x128 block, set True if any True in the block.
            workloads_per_query_block = get_num_computation_block_per_query_block(
                bitfield_attention_mask
            )
            workloads_per_query_block, indices = torch.sort(
                workloads_per_query_block, stable=True, descending=True
            )
            # (total workload, sp_rank)
            loads_per_rank: list[tuple[int, int]] = []
            for i in range(sp_size):
                heapq.heappush(loads_per_rank, (0, i))

            local_seq_lens: list[list[int]] = [[] for _ in range(sp_size)]
            offsets: list[list[np.ndarray]] = [[] for _ in range(sp_size)]

            for batch_index in range(batch_size):
                workloads_per_query_block_per_batch = workloads_per_query_block[
                    batch_index
                ]

                num_tokens_per_block = []
                remaining_tokens = seqlens[batch_index].item()
                for _ in range(len(workloads_per_query_block_per_batch)):
                    num_tokens = min(remaining_tokens, block_size)
                    num_tokens_per_block.append(num_tokens)
                    remaining_tokens -= num_tokens

                seq_lens_per_batch_per_rank = [0 for _ in range(sp_size)]
                assignments_per_rank = [[] for _ in range(sp_size)]
                # Iterate over the blocks and assign them to the process with the least load
                for i, block_workload in zip(
                    indices[batch_index].numpy(force=True),
                    workloads_per_query_block_per_batch,
                ):
                    load, rank = heapq.heappop(loads_per_rank)

                    heapq.heappush(
                        loads_per_rank,
                        (load + block_workload.item(), rank),
                    )

                    seq_lens_per_batch_per_rank[rank] += num_tokens_per_block[i]
                    assignments_per_rank[rank].append(i)

                for rank, assignments in enumerate(assignments_per_rank):
                    assignments.sort()
                    if assignments:
                        assignments = np.concatenate(
                            [
                                np.arange(
                                    sum(num_tokens_per_block[:i]),
                                    sum(num_tokens_per_block[:i])
                                    + num_tokens_per_block[i],
                                )
                                for i in assignments
                            ]
                        )
                    else:
                        assert seq_lens_per_batch_per_rank[rank] == 0

                    local_seq_lens[rank].append(seq_lens_per_batch_per_rank[rank])
                    offsets[rank].append(assignments)

            ContextParallelBatchSplitUtils.set_context_parallel_sequence_lengths_cache(
                local_seq_lens, offsets
            )
            local_seq_lens, offsets = (
                ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache(
                    sp_rank
                )
            )

        if is_label:
            # shift one position so that tokens < n predict n
            # shift should only happen for text tokens, so first
            # find the non-pad token and shift from there.
            batch = nn.functional.pad(batch, (0, 1), value=-100)[..., 1:]

        split_batches = []
        for i in range(batch_size):
            split_batches.append(batch[i, offsets[i]])

        return (
            torch.nested.as_nested_tensor(split_batches, device=batch.device)
            .to_padded_tensor(padding=0.0)
            .contiguous(),
            local_seq_lens,
            offsets,
        )

    @classmethod
    def _split_batch_random(
        cls: ContextParallelBatchSplitUtils,
        batch: torch.Tensor,
        bitfield_attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        is_label: bool = False,
    ) -> torch.Tensor:
        """
        split tokens randomly. If the number of tokens is large it is magically balanced.

        This uses a hash function to assign tokens to processes. The hash function is
        `hash(token_index + random_offset) % sp_size == sp_rank`, where `hash()` is a simple
        linear hash function.
        To ensure even distribution, a and mod (sp_size) should be coprime, i.e. their GCD is 1.
        """
        raise NotImplementedError
        # sp_size = dist.get_world_size(sp_group)
        # sp_rank = dist.get_rank(sp_group)
        # if sp_size == 1:
        #     return batch

        # batch_size, seq_len = bitfield_attention_mask.shape

        # if cls.split_batch_cache is not None:
        #     assert cls.split_batch_cache.shape == (seq_len,), (
        #         f"Random split cache shape {cls.split_batch_cache.shape} "
        #         f"does not match the sequence length {seq_len}"
        #     )
        #     assignments = cls.split_batch_cache
        # else:

        #     def generate_coprime_a(p, mod):
        #         while True:
        #             a = np.random.randint(1, p)
        #             if np.gcd(a, mod) == 1:
        #                 return a

        #     # Hash function parameters
        #     p = 2**31  # Modulus
        #     a = generate_coprime_a(p, sp_size)  # multiplier
        #     b = np.random.randint(0, p)  # increment
        #     offset = np.random.randint(0, p)

        #     token_indices = np.arange(seq_len, dtype=np.int64)  # shape: [seq_len]

        #     # Compute the hash for each index
        #     hash_values = (
        #         a * ((token_indices + offset) % p) + b
        #     ) % p  # shape: [seq_len]

        #     # Determine assignment based on hash modulo 'mod'
        #     assignments = (hash_values % sp_size) == sp_rank  # shape: [seq_len]

        #     # Cache assignments
        #     cls.split_batch_cache = assignments

        # # Extract the indices assigned to this process
        # assigned_indices = np.flatnonzero(assignments)  # shape: [num_assigned_indices]
        # assigned_indices = torch.as_tensor(
        #     assigned_indices, dtype=torch.long, device=batch.device
        # )

        # # Create a slice to index of selected tokens in seq_dim
        # slices = [slice(None)] * batch.dim()
        # seq_dim = 1
        # slices[seq_dim] = (
        #     assigned_indices  # replace only the seq_dim with assigned indices
        # )

        # # Use advanced indexing with slices to select the tokens
        # return batch[slices].contiguous()
