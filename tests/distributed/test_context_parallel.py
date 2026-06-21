"""Tests for context-parallel splitters.

Uses a 2-rank gloo setup to verify that each splitter type correctly partitions
the sequence across ranks, with no overlap and full coverage.
"""
import unittest

import torch
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase
from cornstarch.distributed.context_parallel.splitters import (
    MakespanMinContextParallelSplitter,
    UniformContextParallelSplitter,
    ZigzagContextParallelSplitter,
)


def _make_mask(batch: int, seq: int) -> torch.Tensor:
    """Dense all-ones mask for splitter testing."""
    return torch.ones(batch, seq, dtype=torch.float32)


class TestUniformSplitter(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _get_cp_group(self) -> dist.ProcessGroup:
        return dist.group.WORLD

    def test_coverage_and_no_overlap(self):
        """Union of per-rank indices covers the full sequence with no duplicates."""
        mask = _make_mask(2, 128)
        cp_group = self._get_cp_group()
        splitter = UniformContextParallelSplitter()
        offsets_per_rank = splitter.compute_offsets(mask, cp_group)

        rank = dist.get_rank(cp_group)
        local_offsets = offsets_per_rank[rank]

        # Gather all local offsets to rank 0.
        all_offsets = [torch.zeros_like(local_offsets) for _ in range(self.world_size)]
        dist.all_gather(all_offsets, local_offsets.clone())

        if rank == 0:
            combined = torch.cat(all_offsets)
            self.assertEqual(combined.numel(), 128)
            self.assertEqual(combined.unique().numel(), 128)

    def test_split_returns_local_slice(self):
        """split() should return exactly this rank's assigned positions."""
        mask = _make_mask(2, 64)
        cp_group = self._get_cp_group()
        splitter = UniformContextParallelSplitter()
        splitter.compute_offsets(mask, cp_group)

        x = torch.arange(64).unsqueeze(0).expand(2, -1).float()  # (2, 64)
        local = splitter.split(x, cp_group)

        rank = dist.get_rank(cp_group)
        expected_len = 64 // self.world_size
        self.assertEqual(local.shape, (2, expected_len))

        # Verify values correspond to the right half of the sequence.
        expected_start = rank * expected_len
        self.assertTrue(
            torch.all(local[0] == torch.arange(expected_start, expected_start + expected_len).float())
        )


class TestZigzagSplitter(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _get_cp_group(self) -> dist.ProcessGroup:
        return dist.group.WORLD

    def test_coverage_and_no_overlap(self):
        """Zigzag assignment should also cover the full sequence exactly once."""
        mask = _make_mask(1, 64)
        cp_group = self._get_cp_group()
        splitter = ZigzagContextParallelSplitter()
        offsets_per_rank = splitter.compute_offsets(mask, cp_group)

        rank = dist.get_rank(cp_group)
        local_offsets = offsets_per_rank[rank]

        # Each rank should have 32 positions.
        self.assertEqual(local_offsets.numel(), 32)

        all_offsets = [torch.zeros_like(local_offsets) for _ in range(self.world_size)]
        dist.all_gather(all_offsets, local_offsets.clone())

        if rank == 0:
            combined = torch.cat(all_offsets)
            self.assertEqual(combined.numel(), 64)
            self.assertEqual(combined.unique().numel(), 64)

    def test_rank0_gets_first_and_last_chunk(self):
        """For 2 ranks, rank 0 should own the first and last quarter (zigzag pairing)."""
        mask = _make_mask(1, 64)
        cp_group = self._get_cp_group()
        splitter = ZigzagContextParallelSplitter()
        offsets_per_rank = splitter.compute_offsets(mask, cp_group)

        rank = dist.get_rank(cp_group)
        if rank == 0:
            # 4 halves: [0..15], [16..31], [32..47], [48..63]
            # Rank 0 pairs halves[0] + halves[3] = [0..15] ∪ [48..63]
            expected = torch.cat([torch.arange(0, 16), torch.arange(48, 64)])
            self.assertTrue(torch.equal(offsets_per_rank[0], expected))


class TestMakespanMinSplitter(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _get_cp_group(self) -> dist.ProcessGroup:
        return dist.group.WORLD

    def test_coverage_and_no_overlap_2d(self):
        """Makespan-min (2D mask) should cover the full sequence with no duplicates."""
        mask = _make_mask(2, 256)
        cp_group = self._get_cp_group()
        splitter = MakespanMinContextParallelSplitter(block_size=32)
        offsets_per_rank = splitter.compute_offsets(mask, cp_group)

        rank = dist.get_rank(cp_group)
        local_offsets = offsets_per_rank[rank]

        # Gather all offsets and verify full coverage.
        max_len = max(o.numel() for o in offsets_per_rank)
        # Pad to same length for all_gather.
        padded = torch.full((max_len,), -1, dtype=torch.long)
        padded[: local_offsets.numel()] = local_offsets

        all_offsets = [torch.zeros(max_len, dtype=torch.long) for _ in range(self.world_size)]
        dist.all_gather(all_offsets, padded)

        if rank == 0:
            combined = torch.cat([o[o >= 0] for o in all_offsets])
            self.assertEqual(combined.numel(), 256)
            self.assertEqual(combined.unique().numel(), 256)


if __name__ == "__main__":
    unittest.main()
