"""Tests for context-parallel splitters.

Uses a 2-rank gloo setup to verify that each splitter type correctly partitions
the sequence across ranks, with no overlap and full coverage.
"""
import unittest

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from tests.distributed.distributed_base import GlooDistributedTestBase
from cornstarch.distributed.context_parallel.splitters import (
    MakespanMinContextParallelSplitter,
    UniformContextParallelSplitter,
    ZigzagContextParallelSplitter,
)
from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.distributed.context_parallel import apply_context_parallel
from cornstarch.models import from_hf_config
from tests.model.model_configs import llama_config


def _make_mask(batch: int, seq: int) -> torch.Tensor:
    """Dense all-ones mask for splitter testing."""
    return torch.ones(batch, seq, dtype=torch.float32)


def test_apply_context_parallel_updates_leaf_configs_and_isolates_modules() -> None:
    """Real HF attention leaves dispatch to their own module-bound CP callable."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    config = llama_config()
    first = from_hf_config(config, model_kind="language", attn_implementation="eager")
    second = from_hf_config(config, model_kind="language", attn_implementation="eager")
    first_group, second_group = object(), object()

    first_key = apply_context_parallel(first, first_group, causal=True)
    second_key = apply_context_parallel(second, second_group, causal=False)

    assert first_key != second_key
    assert first.decoder_layers[0].self_attn.config._attn_implementation == first_key
    assert second.decoder_layers[0].self_attn.config._attn_implementation == second_key
    assert ALL_ATTENTION_FUNCTIONS[first_key].keywords == {
        "cp_group": first_group,
        "causal": True,
    }
    assert ALL_ATTENTION_FUNCTIONS[second_key].keywords == {
        "cp_group": second_group,
        "causal": False,
    }


class _CPTextDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.arange(8) + index * 10
        return {"input_ids": ids, "labels": ids.clone()}


class TestContextParallelDataloader(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_builds_global_causal_metadata_before_split(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model = from_hf_config(
            llama_config(), model_kind="language", attn_implementation="eager"
        )
        model.set_random_init()
        plan = ParallelizationPlan(global_ranks=[0, 1])
        plan.parallelize(
            model,
            ParallelConfig(
                context_parallel_size=2,
                context_parallel_splitter=UniformContextParallelSplitter(),
                data_parallel_size=1,
            ),
        )
        ctx = plan.materialize("cpu", dtype=torch.float32)
        attention_key = model.decoder_layers[0].self_attn.config._attn_implementation
        self.assertTrue(ALL_ATTENTION_FUNCTIONS[attention_key].keywords["causal"])
        batch = next(iter(ctx.prepare_dataloader(_CPTextDataset(), batch_size=2)))[0]

        rank = dist.get_rank()
        offsets = torch.arange(rank * 4, (rank + 1) * 4)
        expected_ids = batch["cp_global_input_ids"].index_select(1, offsets)
        self.assertTrue(torch.equal(batch["input_ids"], expected_ids))
        self.assertTrue(
            torch.equal(batch["position_ids"], offsets.unsqueeze(0).expand(2, -1))
        )

        global_labels = batch["cp_global_input_ids"]
        global_shift = torch.empty_like(global_labels)
        global_shift[:, :-1] = global_labels[:, 1:]
        global_shift[:, -1] = -100
        self.assertTrue(
            torch.equal(batch["shift_labels"], global_shift.index_select(1, offsets))
        )
        self.assertEqual(int(batch["num_items_in_batch"]), 14)


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
