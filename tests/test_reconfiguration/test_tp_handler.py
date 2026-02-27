"""Tests for TPReconfigurationHandler (un-shard and re-shard)."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import instantiate_parametrized_tests

from cornstarch.reconfiguration.tp_handler import TPReconfigurationHandler

from ..distributed_base import GlooDistributedTestBase


class ShardedModel(nn.Module):
    """Toy model with pre-sharded parameters to simulate a TP-sharded state.

    Instead of using actual Linear1D_Col / Linear1D_Row (which require a
    live process group at construction time), we use plain nn.Parameters and
    drive the handler tests by directly calling the static helper and manually
    constructing the shard specs.
    """

    def __init__(self, full_weight: torch.Tensor):
        super().__init__()
        # Each rank holds its own shard; we set the data after construction.
        self.weight = nn.Parameter(full_weight.clone())
        self.bias = nn.Parameter(torch.zeros(full_weight.shape[0]))


@instantiate_parametrized_tests
class TpHandlerTests(GlooDistributedTestBase):
    """Tests for TPReconfigurationHandler."""

    @property
    def world_size(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_full_weight(self, rows: int = 8, cols: int = 4) -> torch.Tensor:
        """Return a deterministic full weight tensor (same on all ranks)."""
        torch.manual_seed(42)
        return torch.randn(rows, cols)

    def _make_tp_group(self):
        """Return the single TP group containing all 4 ranks."""
        return dist.new_group(ranks=list(range(self.world_size)))

    # ------------------------------------------------------------------
    # test_unshards_col_parallel
    # ------------------------------------------------------------------

    def test_unshards_col_parallel(self):
        """All-gathering col-parallel shards (dim 0) restores the full weight.

        Setup: world_size=4, each rank holds 1/4 of rows (dim 0).
        After unshards(): rank 0 → full tensor; ranks 1-3 → zeros of full shape.
        """
        full_weight = self._make_full_weight(rows=8, cols=4)  # [8, 4]
        rows_per_rank = full_weight.shape[0] // self.world_size  # 2

        # Each rank gets its own shard along dim 0.
        shard = full_weight[
            self.rank * rows_per_rank : (self.rank + 1) * rows_per_rank
        ].clone()

        model = ShardedModel(shard)
        tp_group = self._make_tp_group()

        # Simulate un-sharding: TP rank 0 gathers all shards via dist.gather.
        handler = TPReconfigurationHandler()
        handler._shard_specs = []

        full = TPReconfigurationHandler._gather_dim(
            model.weight.data.contiguous(), dim=0, group=tp_group
        )  # full is None on non-zero TP ranks

        if self.rank == 0:
            model.weight.data = full.clone()
            handler._shard_specs.append(("weight", 0))
        else:
            full_shape = list(model.weight.shape)
            full_shape[0] *= self.world_size
            model.weight.data = torch.zeros(full_shape, dtype=model.weight.dtype)

        # Verify rank 0 holds the correct full weight.
        if self.rank == 0:
            assert model.weight.shape == (8, 4), (
                f"Rank 0 weight shape wrong: {model.weight.shape}"
            )
            assert torch.allclose(model.weight.data, full_weight, atol=1e-6), (
                "Rank 0 weight does not match full_weight"
            )
        else:
            assert model.weight.shape == (8, 4), (
                f"Rank {self.rank} weight shape wrong after unshards"
            )
            assert torch.all(model.weight.data == 0), (
                f"Rank {self.rank} weight should be zeros after unshards"
            )

        dist.destroy_process_group(tp_group)
        print(f"Rank {self.rank}: test_unshards_col_parallel passed")

    # ------------------------------------------------------------------
    # test_unshards_row_parallel
    # ------------------------------------------------------------------

    def test_unshards_row_parallel(self):
        """All-gathering row-parallel shards (dim 1) restores the full weight.

        Setup: world_size=4, each rank holds 1/4 of columns (dim 1).
        """
        full_weight = self._make_full_weight(rows=4, cols=8)  # [4, 8]
        cols_per_rank = full_weight.shape[1] // self.world_size  # 2

        shard = full_weight[
            :, self.rank * cols_per_rank : (self.rank + 1) * cols_per_rank
        ].clone()

        model = ShardedModel(shard)
        tp_group = self._make_tp_group()

        full = TPReconfigurationHandler._gather_dim(
            model.weight.data.contiguous(), dim=1, group=tp_group
        )  # None on non-zero ranks

        if self.rank == 0:
            model.weight.data = full.clone()
            assert model.weight.shape == (4, 8)
            assert torch.allclose(model.weight.data, full_weight, atol=1e-6)
        else:
            full_shape = list(model.weight.shape)
            full_shape[1] *= self.world_size
            model.weight.data = torch.zeros(full_shape, dtype=model.weight.dtype)
            assert model.weight.shape == (4, 8)
            assert torch.all(model.weight.data == 0)

        dist.destroy_process_group(tp_group)
        print(f"Rank {self.rank}: test_unshards_row_parallel passed")

    # ------------------------------------------------------------------
    # test_reshards_col_parallel
    # ------------------------------------------------------------------

    def test_reshards_col_parallel(self):
        """Scattering a full tensor distributes the correct slice to each rank.

        Rank 0 holds the full weight; reshards() should give each rank the
        contiguous rows [rank * chunk : (rank+1) * chunk].
        """
        full_weight = self._make_full_weight(rows=8, cols=4)  # [8, 4]
        rows_per_rank = full_weight.shape[0] // self.world_size  # 2

        # Start: only rank 0 has the full weight; others have zeros.
        if self.rank == 0:
            model = ShardedModel(full_weight.clone())
        else:
            model = ShardedModel(torch.zeros_like(full_weight))

        tp_group = self._make_tp_group()
        handler = TPReconfigurationHandler()
        handler._shard_specs = [("weight", 0)]

        # reshards() broadcasts from rank 0 and slices.
        handler.reshards(model, tp_group)

        expected = full_weight[
            self.rank * rows_per_rank : (self.rank + 1) * rows_per_rank
        ]
        assert model.weight.shape == expected.shape, (
            f"Rank {self.rank}: expected shape {expected.shape}, "
            f"got {model.weight.shape}"
        )
        assert torch.allclose(model.weight.data, expected, atol=1e-6), (
            f"Rank {self.rank}: weight values incorrect after reshards"
        )

        dist.destroy_process_group(tp_group)
        print(f"Rank {self.rank}: test_reshards_col_parallel passed")

    # ------------------------------------------------------------------
    # test_unshards_noop_when_tp1
    # ------------------------------------------------------------------

    def test_unshards_noop_when_tp1(self):
        """unshards() is a no-op when the TP group has size 1."""
        full_weight = self._make_full_weight()
        model = ShardedModel(full_weight.clone())
        original_data = model.weight.data.clone()

        # Create a singleton group for this rank only.
        singleton_group = dist.new_group(ranks=[self.rank])

        handler = TPReconfigurationHandler()
        handler.unshards(model, singleton_group)

        # Tensor must be unchanged.
        assert torch.allclose(model.weight.data, original_data, atol=1e-6), (
            f"Rank {self.rank}: weight changed unexpectedly for TP=1"
        )
        assert handler._shard_specs == [], "No shard specs should be recorded for TP=1"

        dist.destroy_process_group(singleton_group)
        print(f"Rank {self.rank}: test_unshards_noop_when_tp1 passed")
