from __future__ import annotations

import heapq
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator

from cornstarch.kernel.attention import BLOCK_M, BLOCK_N
from cornstarch.shardformer.shard.shard_config import ContextParallelDistributionMode


class ContextParallelBatchSplitUtils:
    # Context parallel cache for offsets per rank.
    # Each tensor in the lists is a tensor of offsets for a rank.
    # Each tensor is a 1D tensor of shape (local_seqlen).
    context_parallel_offsets_cache: Optional[list[torch.Tensor]] = None

    @classmethod
    def clear_cache(cls: ContextParallelBatchSplitUtils):
        cls.context_parallel_offsets_cache = None

    @classmethod
    def set_context_parallel_offsets_cache(
        cls: ContextParallelBatchSplitUtils,
        offsets_per_rank: list[np.ndarray] | list[torch.Tensor],
        device: torch.device = None,
    ):
        if device is None:
            device = get_accelerator().get_current_device()

        if isinstance(offsets_per_rank[0], np.ndarray):
            offsets_per_rank = [
                torch.tensor(offsets, dtype=torch.long, device=device)
                for offsets in offsets_per_rank
            ]

        cls.context_parallel_offsets_cache = offsets_per_rank

    @classmethod
    def get_seqlen_per_rank(
        cls: ContextParallelBatchSplitUtils, device: torch.device = None
    ) -> Optional[torch.Tensor]:
        if device is None:
            device = get_accelerator().get_current_device()

        if cls.context_parallel_offsets_cache is None:
            return None

        return torch.tensor(
            [
                len(offsets_per_rank)
                for offsets_per_rank in cls.context_parallel_offsets_cache
            ],
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def get_context_parallel_offsets_cache(
        cls: ContextParallelBatchSplitUtils, sp_rank: int = -1
    ) -> Optional[torch.Tensor]:
        assert sp_rank == -1 or sp_rank >= 0
        if cls.context_parallel_offsets_cache is None:
            return None

        offsets = cls.context_parallel_offsets_cache
        if sp_rank == -1:
            return offsets
        else:
            return offsets[sp_rank]

    @classmethod
    def create_context_parallel_split(
        cls: ContextParallelBatchSplitUtils,
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        dist_mode: ContextParallelDistributionMode,
        **kwargs,
    ):
        """
        Based on attention_mask and distribution mode,
        creates a context parallel split across ranks and set offsets cache.
        After setting the cache, states (hidden_states and labels) can be split using the offsets.
        """
        if cls.context_parallel_offsets_cache is not None:
            return

        assert attention_mask is not None and attention_mask.ndim in [2, 3]

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        if sp_size == 1:
            return

        if attention_mask.ndim == 2:
            batch_size, seq_len = attention_mask.shape
        else:
            batch_size, seq_len = attention_mask.shape[:2]

        if seq_len < 128 * sp_size:
            return

        if cls.context_parallel_offsets_cache is None:
            if dist_mode == ContextParallelDistributionMode.UNIFORM:
                cls.create_context_parallel_split_uniform(
                    attention_mask, sp_group, **kwargs
                )
            elif dist_mode == ContextParallelDistributionMode.ZIGZAG:
                cls.create_context_parallel_split_zigzag(
                    attention_mask, sp_group, **kwargs
                )
            elif dist_mode == ContextParallelDistributionMode.MAKESPAN_MIN:
                cls.create_context_parallel_split_makespan_minimization(
                    attention_mask, sp_group, **kwargs
                )

    @classmethod
    def create_context_parallel_split_uniform(
        cls: ContextParallelBatchSplitUtils,
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        **kwargs,
    ):
        assert attention_mask.ndim in [2, 3]
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        if cls.context_parallel_offsets_cache is not None:
            return

        batch_size, seq_len = attention_mask.shape[:2]
        chunked_offsets = np.array_split(np.arange(seq_len), sp_size)
        cls.set_context_parallel_offsets_cache(chunked_offsets)

    @classmethod
    def create_context_parallel_split_zigzag(
        cls: ContextParallelBatchSplitUtils,
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        **kwargs,
    ):
        assert attention_mask.ndim in [2, 3]
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        batch_size, seq_len = attention_mask.shape[:2]
        # Divide indices into 2*sp_size chunks.
        chunked_offsets = np.array_split(np.arange(seq_len), sp_size * 2)
        # Merge the first half and the second half in reverse order.
        chunked_offsets = [
            np.concatenate([chunked_offsets[i], chunked_offsets[-i - 1]])
            for i in range(sp_size)
        ]
        cls.set_context_parallel_offsets_cache(chunked_offsets)

    @classmethod
    def create_context_parallel_split_makespan_minimization(
        cls: ContextParallelBatchSplitUtils,
        attention_mask: torch.Tensor,
        sp_group: dist.ProcessGroup,
        attention_block_exec_time: float = 1.0,
        linear_block_exec_time: float = 1.0,
    ):
        assert attention_mask.ndim == 3
        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        batch_size, seqlen_q, seqlen_k = attention_mask.shape
        original_seqlen_q = seqlen_q
        original_seqlen_k = seqlen_k

        if seqlen_q % BLOCK_M != 0:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(
                        batch_size,
                        BLOCK_M - seqlen_q % BLOCK_M,
                        seqlen_k,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            seqlen_q = attention_mask.shape[1]

        if seqlen_k % BLOCK_N != 0:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(
                        batch_size,
                        seqlen_q,
                        BLOCK_N - seqlen_k % BLOCK_N,
                        device=attention_mask.device,
                    ),
                ],
                dim=2,
            )
            seqlen_k = attention_mask.shape[2]

        # Compress the attention mask.
        num_blocks_to_compute_in_attention = (
            attention_mask.reshape(
                attention_mask.shape[0],
                seqlen_q // BLOCK_M,
                BLOCK_M,
                seqlen_k // BLOCK_N,
                BLOCK_N,
            )
            .any(dim=(2, 4))
            .sum(dim=2)
        )

        # Blocks with the same row-index across batches are merged into one block.
        # workloads_per_block: (num_blocks)
        workloads_per_block = (
            num_blocks_to_compute_in_attention * attention_block_exec_time
            + linear_block_exec_time
        ).sum(dim=0)
        workloads_per_block, indices = torch.sort(
            workloads_per_block, stable=True, descending=True
        )

        # (total_workload, sp_rank)
        loads_per_rank: list[tuple[int, int]] = []
        for i in range(sp_size):
            heapq.heappush(loads_per_rank, (0, i))

        # offsets per rank
        assigned_block_indices_per_rank: list[np.ndarray] = [
            np.array([], dtype=np.int64) for _ in range(sp_size)
        ]

        for i, block_workload in zip(
            indices.numpy(force=True), workloads_per_block.numpy(force=True)
        ):
            # Pick the rank with the minimum workloads
            load, rank = heapq.heappop(loads_per_rank)

            # Assign this row block to the rank
            heapq.heappush(
                loads_per_rank,
                (load + block_workload.item(), rank),
            )

            assigned_block_indices_per_rank[rank] = np.append(
                assigned_block_indices_per_rank[rank], i
            )

        # Sorted assigned block indices
        assigned_block_indices_per_rank = [
            np.sort(indices) for indices in assigned_block_indices_per_rank
        ]

        # expand assigned_block_indices_per_rank
        # for each element in each np.ndarray, expand it to (i, i+1, ..., i+BLOCK_M-1)
        chunked_offsets = [
            (indices[:, None] * BLOCK_M + np.arange(BLOCK_M)).ravel()
            for indices in assigned_block_indices_per_rank
        ]

        # Remove overflowed indices
        chunked_offsets = [
            offsets[offsets < original_seqlen_k] for offsets in chunked_offsets
        ]

        cls.set_context_parallel_offsets_cache(chunked_offsets)

    @classmethod
    def split_batch(
        cls: ContextParallelBatchSplitUtils,
        batch: torch.Tensor,
        sp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        assert cls.context_parallel_offsets_cache is not None, "Offsets must be set."

        sp_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)

        batch_size, seq_len = batch.shape[:2]
        if sp_size == 1 or seq_len < 128 * sp_size:
            return batch

        return batch[:, cls.context_parallel_offsets_cache[sp_rank]]

    @classmethod
    def shuffle_attention_mask(
        cls: ContextParallelBatchSplitUtils,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        When kvs are gathered, they are not reordered in the original order,
        instead simply concatenated.
        This function shuffles the attention mask's columns to match the order of the kvs.
        """

        def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(perm.shape[0], device=perm.device)
            return inv

        merged_offsets = inverse_permutation(
            torch.cat(cls.context_parallel_offsets_cache, dim=0)
        )

        return mask[:, :, merged_offsets].contiguous()
