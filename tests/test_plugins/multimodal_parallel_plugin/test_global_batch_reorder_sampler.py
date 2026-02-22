"""Unit tests for GlobalBatchReorderSampler.

These tests exercise the sampler in isolation — no distributed process group
is required because all logic is local Python.
"""
from __future__ import annotations

from typing import List

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cornstarch.plugin.multimodal_parallel_plugin.global_batch_reorder_sampler import (
    GlobalBatchReorderSampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dataset(n: int) -> TensorDataset:
    return TensorDataset(torch.arange(n, dtype=torch.float32))


def collect_all_ranks(
    dataset,
    num_replicas: int,
    global_batch_size: int,
    *,
    metadata_fn=None,
    batch_reorder_fn=None,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    epoch: int = 0,
) -> List[List[int]]:
    """Return the flat index list yielded by each rank's sampler for one epoch.

    The sampler yields one ``List[int]`` per step; this helper flattens those
    into a single list per rank so that index-level assertions remain concise.
    """
    result = []
    for rank in range(num_replicas):
        sampler = GlobalBatchReorderSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            global_batch_size=global_batch_size,
            metadata_fn=metadata_fn,
            batch_reorder_fn=batch_reorder_fn,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        sampler.set_epoch(epoch)
        result.append([idx for batch in sampler for idx in batch])
    return result


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_neither_fn_uses_identity(self):
        # No fn provided — must construct without error and yield a valid partition.
        ds = make_dataset(8)
        per_rank = collect_all_ranks(ds, num_replicas=2, global_batch_size=4)
        all_seen = sorted(idx for rank_indices in per_rank for idx in rank_indices)
        assert all_seen == list(range(8))
        # Contiguous: rank 0 gets [0,1] and [4,5]; rank 1 gets [2,3] and [6,7]
        assert per_rank[0] == [0, 1, 4, 5]
        assert per_rank[1] == [2, 3, 6, 7]

    def test_both_fn_raises(self):
        ds = make_dataset(10)
        with pytest.raises(ValueError, match="not both"):
            GlobalBatchReorderSampler(
                ds,
                num_replicas=2,
                rank=0,
                global_batch_size=4,
                metadata_fn=lambda i: float(i),
                batch_reorder_fn=lambda idxs: idxs,
            )

    def test_global_batch_size_not_divisible_raises_without_batch_reorder_fn(self):
        ds = make_dataset(10)
        with pytest.raises(ValueError, match="divisible"):
            GlobalBatchReorderSampler(
                ds,
                num_replicas=3,
                rank=0,
                global_batch_size=5,  # 5 % 3 != 0, no batch_reorder_fn
            )

    def test_global_batch_size_not_divisible_ok_with_batch_reorder_fn(self):
        # When batch_reorder_fn is provided, unequal global_batch_size is allowed.
        ds = make_dataset(10)
        reorder_fn = lambda idxs: [idxs[:3], idxs[3:4], idxs[4:]]  # 3+1+1=5
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=3,
            rank=0,
            global_batch_size=5,  # not divisible by 3 — OK here
            batch_reorder_fn=reorder_fn,
        )
        assert sampler is not None

    def test_invalid_rank_raises(self):
        ds = make_dataset(10)
        with pytest.raises(ValueError, match="rank must be in"):
            GlobalBatchReorderSampler(
                ds,
                num_replicas=2,
                rank=2,
                global_batch_size=4,
                metadata_fn=lambda i: float(i),
            )

    def test_negative_rank_raises(self):
        ds = make_dataset(10)
        with pytest.raises(ValueError, match="rank must be in"):
            GlobalBatchReorderSampler(
                ds,
                num_replicas=2,
                rank=-1,
                global_batch_size=4,
                metadata_fn=lambda i: float(i),
            )


# ---------------------------------------------------------------------------
# __len__ tests
# ---------------------------------------------------------------------------


class TestLen:
    def test_len_exact_divisible(self):
        # 12 samples, global_bs=6 → 2 global batches
        ds = make_dataset(12)
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=3,
            rank=0,
            global_batch_size=6,
            metadata_fn=lambda i: float(i),
        )
        assert len(sampler) == 2  # 2 global batches

    def test_len_drop_last(self):
        # 13 samples, global_bs=6 → 2 full global batches (12), 1 dropped
        ds = make_dataset(13)
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=3,
            rank=0,
            global_batch_size=6,
            metadata_fn=lambda i: float(i),
            drop_last=True,
        )
        assert len(sampler) == 2  # 2 global batches

    def test_len_pad(self):
        # 13 samples, global_bs=6 → pad to 18 → 3 global batches
        ds = make_dataset(13)
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=3,
            rank=0,
            global_batch_size=6,
            metadata_fn=lambda i: float(i),
            drop_last=False,
        )
        assert len(sampler) == 3  # 3 global batches


# ---------------------------------------------------------------------------
# metadata_fn path: sort-by-descending-cost
# ---------------------------------------------------------------------------


class TestMetadataFn:
    def test_within_global_batch_sorted_descending(self):
        """Within each global batch, rank 0 must get the highest-cost samples."""
        n, num_replicas, global_bs = 12, 3, 6
        # Cost = identity (cost[i] == i)
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
        )
        # global batches: [0..5] → sorted desc [5,4,3,2,1,0]
        #                 [6..11] → sorted desc [11,10,9,8,7,6]
        # rank 0 (per_rank_size=2): [5,4] + [11,10]
        assert per_rank[0] == [5, 4, 11, 10]
        # rank 1: [3,2] + [9,8]
        assert per_rank[1] == [3, 2, 9, 8]
        # rank 2: [1,0] + [7,6]
        assert per_rank[2] == [1, 0, 7, 6]

    def test_no_sample_lost_or_duplicated(self):
        """All samples in each global batch must appear exactly once across ranks."""
        n, num_replicas, global_bs = 12, 3, 6
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
        )
        all_seen = [idx for rank_indices in per_rank for idx in rank_indices]
        assert sorted(all_seen) == list(range(n))

    def test_metadata_fn_called_once_per_sample(self):
        """metadata_fn must be called exactly len(dataset) times during __init__."""
        n = 20
        call_count = []
        ds = make_dataset(n)

        def counting_fn(i: int) -> float:
            call_count.append(i)
            return float(i)

        GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=4,
            metadata_fn=counting_fn,
        )
        assert sorted(call_count) == list(range(n))

    def test_metadata_fn_not_called_during_iteration(self):
        """After __init__, iterating must not trigger metadata_fn again."""
        n = 20
        ds = make_dataset(n)
        init_calls = []

        def counting_fn(i: int) -> float:
            init_calls.append(i)
            return float(i)

        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=4,
            metadata_fn=counting_fn,
        )
        calls_after_init = len(init_calls)
        _ = list(sampler)
        _ = list(sampler)
        assert len(init_calls) == calls_after_init  # no additional calls


# ---------------------------------------------------------------------------
# batch_reorder_fn path
# ---------------------------------------------------------------------------


class TestBatchReorderFn:
    def test_custom_reorder_applied(self):
        """batch_reorder_fn receives the full global batch, assigns per-rank lists."""
        n, num_replicas, global_bs = 8, 2, 4
        # Reverse the global batch then split: rank 0 gets first half, rank 1 second.
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            batch_reorder_fn=lambda idxs: [
                list(reversed(idxs))[:2],
                list(reversed(idxs))[2:],
            ],
        )
        # global batches: [0,1,2,3] → reversed [3,2,1,0] → [[3,2],[1,0]]
        #                 [4,5,6,7] → reversed [7,6,5,4] → [[7,6],[5,4]]
        assert per_rank[0] == [3, 2, 7, 6]
        assert per_rank[1] == [1, 0, 5, 4]

    def test_identity_reorder_gives_standard_partition(self):
        """Identity partition → rank r gets contiguous slice [r*prs:(r+1)*prs]."""
        n, num_replicas, global_bs = 8, 2, 4
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            batch_reorder_fn=lambda idxs: [idxs[:2], idxs[2:]],
        )
        # global batches: [0,1,2,3] → [[0,1],[2,3]]
        #                 [4,5,6,7] → [[4,5],[6,7]]
        assert per_rank[0] == [0, 1, 4, 5]
        assert per_rank[1] == [2, 3, 6, 7]

    def test_no_sample_lost_or_duplicated(self):
        n, num_replicas, global_bs = 12, 3, 6
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            batch_reorder_fn=lambda idxs: [
                list(reversed(idxs))[:2],
                list(reversed(idxs))[2:4],
                list(reversed(idxs))[4:],
            ],
        )
        all_seen = [idx for rank_indices in per_rank for idx in rank_indices]
        assert sorted(all_seen) == list(range(n))

    def test_unequal_split_across_ranks(self):
        """batch_reorder_fn may assign different numbers of samples per rank."""
        n, num_replicas, global_bs = 12, 3, 6
        # rank 0 gets 3, rank 1 gets 2, rank 2 gets 1 per global batch
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            batch_reorder_fn=lambda idxs: [idxs[:3], idxs[3:5], idxs[5:]],
        )
        # 2 global batches of 6 → rank 0: 6 indices, rank 1: 4, rank 2: 2
        assert len(per_rank[0]) == 6
        assert len(per_rank[1]) == 4
        assert len(per_rank[2]) == 2
        # All 12 original indices accounted for
        all_seen = [idx for rank_indices in per_rank for idx in rank_indices]
        assert sorted(all_seen) == list(range(n))


# ---------------------------------------------------------------------------
# drop_last / padding
# ---------------------------------------------------------------------------


class TestDropLastAndPadding:
    def test_drop_last_trims_tail(self):
        # 10 samples, global_bs=6 → 1 full global batch (6), 4 dropped
        n, num_replicas, global_bs = 10, 3, 6
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
            drop_last=True,
        )
        # total_size=6, num_samples per rank=2
        for rank_indices in per_rank:
            assert len(rank_indices) == 2
        # All 6 indices from first global batch [0..5] present across ranks
        all_seen = sorted(idx for rank_indices in per_rank for idx in rank_indices)
        assert all_seen == list(range(6))

    def test_pad_repeats_samples(self):
        # 10 samples, global_bs=6 → pad to 12 (2 full global batches)
        n, num_replicas, global_bs = 10, 3, 6
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
            drop_last=False,
        )
        for rank_indices in per_rank:
            assert len(rank_indices) == 4  # 12 / 3
        # Exactly 12 total entries
        all_seen = [idx for rank_indices in per_rank for idx in rank_indices]
        assert len(all_seen) == 12
        # All are valid dataset indices (may repeat due to padding)
        assert all(0 <= idx < n for idx in all_seen)

    def test_len_matches_iter_len(self):
        # __len__ returns num_global_batches; iterating the sampler yields
        # exactly that many lists.
        for drop_last in (True, False):
            for n in (10, 12, 13):
                ds = make_dataset(n)
                sampler = GlobalBatchReorderSampler(
                    ds,
                    num_replicas=3,
                    rank=0,
                    global_batch_size=6,
                    metadata_fn=lambda i: float(i),
                    drop_last=drop_last,
                )
                batches = list(sampler)
                assert len(sampler) == len(batches)
                # Each yielded element must be a list of per_rank_size indices.
                for batch in batches:
                    assert isinstance(batch, list)
                    assert len(batch) == sampler.per_rank_size


# ---------------------------------------------------------------------------
# Shuffle and set_epoch
# ---------------------------------------------------------------------------


class TestShuffleAndEpoch:
    def test_no_shuffle_deterministic(self):
        """With shuffle=False the order is always the same regardless of epoch."""
        ds = make_dataset(12)
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=4,
            metadata_fn=lambda i: float(i),
            shuffle=False,
        )
        result_e0 = list(sampler)
        sampler.set_epoch(1)
        result_e1 = list(sampler)
        assert result_e0 == result_e1

    def test_shuffle_changes_across_epochs(self):
        """With shuffle=True, different epochs must (almost certainly) differ."""
        ds = make_dataset(100)
        sampler = GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=8,
            metadata_fn=lambda i: float(i),
            shuffle=True,
            seed=42,
        )
        sampler.set_epoch(0)
        result_e0 = list(sampler)
        sampler.set_epoch(1)
        result_e1 = list(sampler)
        assert result_e0 != result_e1

    def test_shuffle_same_across_ranks_same_epoch(self):
        """All ranks must see the same global ordering (different slices of it)."""
        n, num_replicas, global_bs = 24, 3, 6
        per_rank_size = global_bs // num_replicas
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
            shuffle=True,
            seed=7,
            epoch=0,
        )
        # No sample duplicated within any single global batch across ranks
        for global_batch_idx in range(len(per_rank[0]) // per_rank_size):
            gathered = []
            for rank_indices in per_rank:
                s = global_batch_idx * per_rank_size
                gathered.extend(rank_indices[s : s + per_rank_size])
            assert len(gathered) == len(set(gathered)), (
                f"Duplicate in global batch {global_batch_idx}: {gathered}"
            )

    def test_all_indices_covered_with_shuffle(self):
        """With shuffle and drop_last=False all dataset indices appear (with possible repeats)."""
        n, num_replicas, global_bs = 20, 4, 12
        per_rank = collect_all_ranks(
            make_dataset(n),
            num_replicas=num_replicas,
            global_batch_size=global_bs,
            metadata_fn=lambda i: float(i),
            shuffle=True,
            seed=0,
            drop_last=False,
        )
        all_seen = sorted(set(idx for rank_indices in per_rank for idx in rank_indices))
        # Every original index must appear at least once
        assert all_seen == list(range(n))

    def test_default_epoch_is_zero(self):
        """set_epoch(0) and a freshly constructed sampler must yield the same result."""
        ds = make_dataset(12)
        sampler_a = GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=4,
            metadata_fn=lambda i: float(i),
            shuffle=True,
            seed=99,
        )
        sampler_b = GlobalBatchReorderSampler(
            ds,
            num_replicas=2,
            rank=0,
            global_batch_size=4,
            metadata_fn=lambda i: float(i),
            shuffle=True,
            seed=99,
        )
        sampler_b.set_epoch(0)
        assert list(sampler_a) == list(sampler_b)


# ---------------------------------------------------------------------------
# Different DataLoader batch_size per rank
# ---------------------------------------------------------------------------


class TestBatchSamplerDataLoader:
    def test_each_step_yields_per_rank_size_indices(self):
        """When used as batch_sampler, every DataLoader step must yield a
        batch of exactly per_rank_size indices — no cross-batch-boundary
        mixing regardless of which rank is used."""
        n, num_replicas, global_bs = 24, 3, 6
        ds = make_dataset(n)

        for rank in range(num_replicas):
            sampler = GlobalBatchReorderSampler(
                ds,
                num_replicas=num_replicas,
                rank=rank,
                global_batch_size=global_bs,
                metadata_fn=lambda i: float(i),
            )
            dl = DataLoader(ds, batch_sampler=sampler)
            steps = 0
            collected = []
            for batch in dl:
                step_indices = batch[0].long().tolist()
                assert len(step_indices) == sampler.per_rank_size, (
                    f"rank {rank} step {steps}: expected {sampler.per_rank_size} "
                    f"indices per step, got {len(step_indices)}"
                )
                collected.extend(step_indices)
                steps += 1

            assert steps == len(sampler), (
                f"rank {rank}: expected {len(sampler)} steps, got {steps}"
            )
            assert all(0 <= idx < n for idx in collected)

    def test_all_ranks_together_cover_dataset(self):
        """Across all ranks, every dataset index appears exactly once per epoch
        when using the sampler as a batch_sampler."""
        n, num_replicas, global_bs = 24, 3, 6
        ds = make_dataset(n)
        all_seen = []
        for rank in range(num_replicas):
            sampler = GlobalBatchReorderSampler(
                ds,
                num_replicas=num_replicas,
                rank=rank,
                global_batch_size=global_bs,
                metadata_fn=lambda i: float(i),
            )
            dl = DataLoader(ds, batch_sampler=sampler)
            for batch in dl:
                all_seen.extend(batch[0].long().tolist())
        assert sorted(all_seen) == list(range(n))
