from __future__ import annotations

import heapq

import numpy as np
import torch
import torch.distributed as dist

from cornstarch.kernel.bitfield_attention import (
    get_num_computation_block_per_query_block,
)
from cornstarch.kernel.interface import BitfieldUtils
from cornstarch.shardformer.shard.shard_config import ContextParallelDistributionMode


class ContextParallelBatchSplitUtils:
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
        seq_lens = []
        offsets = []

        if BitfieldUtils.sequence_lengths_cache is not None:
            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
            )
            split_batches = [batch[i, offset] for i, offset in enumerate(offsets)]
        else:
            for i in range(batch_size):
                # We chunk batch[i] with dim=0, which equivalent to dim=1 in the original tensor.
                batch_chunks = batch[i, : seqlens[i]].chunk(sp_size, dim=0)
                split_batches.append(batch_chunks[sp_rank])

                chunked_seqlens = [chunk.shape[0] for chunk in batch_chunks]
                seq_lens.append(chunked_seqlens[sp_rank])

                start_offset = sum(chunked_seqlens[:sp_rank])
                offsets.append(
                    np.arange(
                        start_offset,
                        start_offset + chunked_seqlens[sp_rank],
                        dtype=np.int32,
                    )
                )

            BitfieldUtils.set_sequence_lengths_cache(
                seqlens.tolist(), seq_lens, offsets
            )

            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
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
        seq_lens = []
        offsets = []

        if BitfieldUtils.sequence_lengths_cache is not None:
            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
            )
        else:
            for i in range(batch_size):
                seq_len = seqlens[i].item()
                indices = np.arange(seq_len)

                # Divide indices into 2*sp_size chunks.
                num_chunks = 2 * sp_size
                chunks = np.array_split(indices, num_chunks)

                # Each rank gets two chunks: the chunk at position 'sp_rank' and the symmetric one.
                first_chunk = chunks[sp_rank]
                second_chunk = chunks[-sp_rank - 1]

                assignments = np.concatenate([first_chunk, second_chunk])

                seq_lens.append(len(assignments))
                offsets.append(assignments)

            # Cache assignments
            BitfieldUtils.set_sequence_lengths_cache(
                seqlens.tolist(), seq_lens, offsets
            )

            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
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
        if sp_size == 1 or seq_len <= 128 * sp_size:
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

        if BitfieldUtils.sequence_lengths_cache is not None:
            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
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

            # sequence length assigned to this process per batch (shape: [batch_size])
            seq_lens = []
            offsets = []

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

                seq_lens_per_batch = 0
                assignments = []
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
                    if rank == sp_rank:
                        seq_lens_per_batch += num_tokens_per_block[i]
                        assignments.append(i)

                assignments.sort()
                if assignments:
                    assignments = np.concatenate(
                        [
                            np.arange(
                                sum(num_tokens_per_block[:i]),
                                sum(num_tokens_per_block[:i]) + num_tokens_per_block[i],
                            )
                            for i in assignments
                        ]
                    )
                else:
                    assert seq_lens_per_batch == 0

                seq_lens.append(seq_lens_per_batch)
                offsets.append(assignments)

            BitfieldUtils.set_sequence_lengths_cache(
                seqlens.tolist(), seq_lens, offsets
            )

            local_seq_lens, seq_lens, offsets = (
                BitfieldUtils.get_sequence_lengths_cache()
            )

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
