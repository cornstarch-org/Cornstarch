"""Context-parallel sequence splitters.

Each splitter computes per-rank token offsets from the attention mask, then
uses those offsets to slice tensors.  Offsets are stored on the instance so
that multiple splitters can coexist (e.g., separate instances per modality).

Usage in a dataloader ``collate_fn`` or training loop::

    splitter = UniformContextParallelSplitter()
    offsets = splitter.compute_offsets(attention_mask, cp_group)
    local_ids       = splitter.split(input_ids, cp_group)
    local_attn_mask = splitter.split(attention_mask, cp_group)
"""
from __future__ import annotations

import heapq
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.distributed as dist


class ContextParallelSplitter(ABC):
    """Abstract base for context-parallel sequence splitters.

    Subclasses implement ``compute_offsets`` which determines, for each CP
    rank, which sequence positions it owns.  After ``compute_offsets`` is
    called once per batch the ``split`` method slices any ``(batch, seq, ...)``
    tensor for the current rank.
    """

    def __init__(self) -> None:
        self._offsets_per_rank: list[torch.Tensor] | None = None

    @abstractmethod
    def compute_offsets(
        self,
        attention_mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        """Compute per-rank index tensors from the attention mask.

        ``attention_mask`` may be 2-D ``(batch, seq_len)`` or 3-D
        ``(batch, seq_q, seq_kv)``.  Returns a list of 1-D ``long`` tensors,
        one per CP rank, each containing the sequence indices assigned to
        that rank.
        """

    def split(
        self,
        batch: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Index ``batch`` along the sequence dimension for the current CP rank.

        ``compute_offsets`` must have been called first for the current batch.
        """
        if self._offsets_per_rank is None:
            raise RuntimeError(
                "compute_offsets() must be called before split()."
            )
        rank = dist.get_rank(cp_group)
        offsets = self._offsets_per_rank[rank]

        if batch.ndim >= 2:
            return batch[:, offsets].contiguous()
        return batch[offsets].contiguous()

    def clear(self) -> None:
        """Reset stored offsets (call between batches if needed)."""
        self._offsets_per_rank = None


class UniformContextParallelSplitter(ContextParallelSplitter):
    """Splits the sequence into equal-length contiguous chunks.

    Rank *k* gets positions ``[k * chunk, (k+1) * chunk)``.  Remainder tokens
    are distributed to the first ranks (numpy ``array_split`` behaviour).
    """

    def compute_offsets(
        self,
        attention_mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        cp_size = dist.get_world_size(cp_group)
        seq_len = attention_mask.shape[1]
        chunks = np.array_split(np.arange(seq_len), cp_size)
        self._offsets_per_rank = [
            torch.tensor(c, dtype=torch.long) for c in chunks
        ]
        return self._offsets_per_rank


class ZigzagContextParallelSplitter(ContextParallelSplitter):
    """Interleaved (zigzag) assignment for causal-attention load balance.

    Divides positions into ``2 × cp_size`` chunks and pairs the first chunk
    with the last, the second with the second-to-last, and so on.  This gives
    every rank a mix of early (compute-heavy with full causal context) and
    late (cheaper) positions, roughly balancing FLOPs across ranks.
    """

    def compute_offsets(
        self,
        attention_mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        cp_size = dist.get_world_size(cp_group)
        seq_len = attention_mask.shape[1]
        halves = np.array_split(np.arange(seq_len), cp_size * 2)
        paired = [
            np.concatenate([halves[i], halves[2 * cp_size - 1 - i]])
            for i in range(cp_size)
        ]
        self._offsets_per_rank = [
            torch.tensor(p, dtype=torch.long) for p in paired
        ]
        return self._offsets_per_rank


class MakespanMinContextParallelSplitter(ContextParallelSplitter):
    """Work-stealing makespan-minimizing assignment.

    Divides the sequence into fixed-size blocks, estimates each block's
    compute cost from the attention mask (number of attended tokens), sorts
    blocks by cost descending, and greedily assigns each block to the rank
    with the lowest accumulated load.  This minimizes the makespan (maximum
    per-rank work) for non-uniform attention patterns such as document masks.

    Works with both 2-D masks ``(batch, seq)`` and 3-D per-query masks
    ``(batch, seq_q, seq_kv)``.
    """

    def __init__(self, block_size: int = 128) -> None:
        super().__init__()
        self._block_size = block_size

    def compute_offsets(
        self,
        attention_mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        assert attention_mask.ndim in (2, 3), (
            "attention_mask must be 2-D (batch, seq) or 3-D (batch, seq_q, seq_kv)"
        )

        if attention_mask.ndim == 2:
            return self._compute_from_2d(attention_mask, cp_group)
        return self._compute_from_3d(attention_mask, cp_group)

    def _assign_blocks_greedy(
        self,
        workloads: np.ndarray,
        cp_size: int,
        seq_len: int,
    ) -> list[torch.Tensor]:
        """Greedy heap-based block-to-rank assignment."""
        sorted_idx = np.argsort(workloads)[::-1]

        heap: list[tuple[int, int]] = [(0, r) for r in range(cp_size)]
        heapq.heapify(heap)

        assigned: list[list[int]] = [[] for _ in range(cp_size)]
        for block_i in sorted_idx:
            load, rank = heapq.heappop(heap)
            assigned[rank].append(int(block_i))
            heapq.heappush(heap, (load + int(workloads[block_i]), rank))

        B = self._block_size
        result: list[torch.Tensor] = []
        for rank_blocks in assigned:
            rank_blocks.sort()
            offsets = np.concatenate(
                [np.arange(b * B, min((b + 1) * B, seq_len)) for b in rank_blocks]
            ) if rank_blocks else np.array([], dtype=np.int64)
            result.append(torch.tensor(offsets, dtype=torch.long))
        return result

    def _compute_from_2d(
        self,
        mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        """Assign blocks by per-block attended-token count (2-D mask)."""
        cp_size = dist.get_world_size(cp_group)
        seq_len = mask.shape[1]
        B = self._block_size

        cpu_mask = mask.cpu().float()
        num_blocks = (seq_len + B - 1) // B
        pad = num_blocks * B - seq_len
        if pad:
            cpu_mask = torch.cat(
                [cpu_mask, torch.zeros(cpu_mask.shape[0], pad)], dim=1
            )
        block_workloads = (
            cpu_mask.reshape(cpu_mask.shape[0], num_blocks, B)
            .sum(dim=(0, 2))
            .numpy()
        )
        self._offsets_per_rank = self._assign_blocks_greedy(
            block_workloads, cp_size, seq_len
        )
        return self._offsets_per_rank

    def _compute_from_3d(
        self,
        mask: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> list[torch.Tensor]:
        """Assign blocks by per-block attended-KV-cell count (3-D mask)."""
        cp_size = dist.get_world_size(cp_group)
        seq_len = mask.shape[1]
        B = self._block_size

        cpu_mask = mask.cpu().float()
        num_blocks = (seq_len + B - 1) // B
        pad_q = num_blocks * B - seq_len
        if pad_q:
            cpu_mask = torch.cat(
                [cpu_mask, torch.zeros(cpu_mask.shape[0], pad_q, cpu_mask.shape[2])],
                dim=1,
            )
        block_workloads = (
            cpu_mask.reshape(cpu_mask.shape[0], num_blocks, B, cpu_mask.shape[2])
            .sum(dim=(0, 2, 3))
            .numpy()
        )
        self._offsets_per_rank = self._assign_blocks_greedy(
            block_workloads, cp_size, seq_len
        )
        return self._offsets_per_rank
