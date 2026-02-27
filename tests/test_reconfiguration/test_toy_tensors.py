"""Toy tensor tests for dynamic reconfiguration."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import instantiate_parametrized_tests

from cornstarch.reconfiguration.data_structures import LayerOwnership
from cornstarch.reconfiguration.executor import ReconfigurationExecutor, TransferPiece

from ..distributed_base import GlooDistributedTestBase


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        # Create some dummy parameters
        self.param1 = nn.Parameter(torch.randn(10, 10))
        self.param2 = nn.Parameter(torch.randn(20, 20))
        self.param3 = nn.Parameter(torch.randn(30, 30))
        self.param4 = nn.Parameter(torch.randn(40, 40))


@instantiate_parametrized_tests
class ToyTensorRedistribution(GlooDistributedTestBase):
    """Test basic tensor redistribution with toy tensors."""

    @property
    def world_size(self) -> int:
        return 4  # 4 ranks for testing

    def test_simple_redistribution(self):
        """Test basic tensor redistribution.

        Source config (PP=2):
        - Rank 0-1: param1, param2
        - Rank 2-3: param3, param4

        Target config (PP=2, different distribution):
        - Rank 0-1: param1, param3
        - Rank 2-3: param2, param4
        """
        # Create model
        model = SimpleModel()

        # Define source ownership
        # Ranks 0-1 hold param1 and param2
        # Ranks 2-3 hold param3 and param4
        source_ownership = {}
        if self.rank in [0, 1]:
            source_ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param1', 'param2'],
                is_placeholder={'param1': False, 'param2': False, 'param3': True, 'param4': True}
            )
        else:  # rank in [2, 3]
            source_ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param3', 'param4'],
                is_placeholder={'param1': True, 'param2': True, 'param3': False, 'param4': False}
            )

        # Gather source ownership from all ranks
        all_source = [None] * self.world_size
        dist.all_gather_object(all_source, source_ownership[self.rank])
        source_ownership = {i: all_source[i] for i in range(self.world_size)}

        # Define target ownership
        # Ranks 0-1 hold param1 and param3
        # Ranks 2-3 hold param2 and param4
        target_ownership = {}
        if self.rank in [0, 1]:
            target_ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param1', 'param3'],
                is_placeholder={'param1': False, 'param2': True, 'param3': False, 'param4': True}
            )
        else:  # rank in [2, 3]
            target_ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param2', 'param4'],
                is_placeholder={'param1': True, 'param2': False, 'param3': True, 'param4': False}
            )

        # Gather target ownership from all ranks
        all_target = [None] * self.world_size
        dist.all_gather_object(all_target, target_ownership[self.rank])
        target_ownership = {i: all_target[i] for i in range(self.world_size)}

        # Save original values for verification
        original_values = {}
        for name, param in model.named_parameters():
            # Copy the original value
            original_values[name] = param.data.clone()

        # Broadcast original values to all ranks so everyone has a reference
        for name in ['param1', 'param2', 'param3', 'param4']:
            # Determine which rank has this parameter in source config
            src_rank = None
            if name in ['param1', 'param2']:
                src_rank = 0  # Use rank 0 as representative
            else:
                src_rank = 2  # Use rank 2 as representative

            # Broadcast from src_rank
            if self.rank == src_rank:
                param = getattr(model, name)
                dist.broadcast(param.data, src=src_rank)
            else:
                # Receive the broadcasted value
                param = getattr(model, name)
                dist.broadcast(param.data, src=src_rank)

            # Update original_values with broadcasted value
            original_values[name] = param.data.clone()

        # Execute reconfiguration
        executor = ReconfigurationExecutor(model)
        executor.execute(source_ownership, target_ownership)

        # Verify redistribution
        # Check that this rank has the correct parameters
        expected_params = target_ownership[self.rank].layer_names
        for param_name in expected_params:
            # This rank should have this parameter
            param = getattr(model, param_name)
            assert param is not None, f"Rank {self.rank} should have {param_name}"
            assert not isinstance(param.data, type(None)), f"Rank {self.rank} should have real data for {param_name}"

            # Verify values match original
            assert torch.allclose(param.data, original_values[param_name], atol=1e-6), \
                f"Rank {self.rank}: {param_name} values don't match after redistribution"

        print(f"Rank {self.rank}: Test passed! Has params: {expected_params}")

    def test_identity_redistribution(self):
        """Test redistribution where source == target (should be no-op).

        Config:
        - Rank 0-1: param1, param2
        - Rank 2-3: param3, param4
        """
        model = SimpleModel()

        # Define ownership (same for source and target)
        ownership = {}
        if self.rank in [0, 1]:
            ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param1', 'param2'],
                is_placeholder={'param1': False, 'param2': False, 'param3': True, 'param4': True}
            )
        else:
            ownership[self.rank] = LayerOwnership(
                rank=self.rank,
                layer_names=['param3', 'param4'],
                is_placeholder={'param1': True, 'param2': True, 'param3': False, 'param4': False}
            )

        # Gather ownership from all ranks
        all_ownership = [None] * self.world_size
        dist.all_gather_object(all_ownership, ownership[self.rank])
        ownership = {i: all_ownership[i] for i in range(self.world_size)}

        # Save original values
        original_values = {}
        expected_params = ownership[self.rank].layer_names
        for param_name in expected_params:
            param = getattr(model, param_name)
            original_values[param_name] = param.data.clone()

        # Execute reconfiguration (should be no-op)
        executor = ReconfigurationExecutor(model)
        executor.execute(ownership, ownership)

        # Verify nothing changed
        for param_name in expected_params:
            param = getattr(model, param_name)
            assert torch.allclose(param.data, original_values[param_name], atol=1e-6), \
                f"Rank {self.rank}: {param_name} values changed unexpectedly"

        print(f"Rank {self.rank}: Identity test passed!")

    def test_tp_shard_to_shard(self):
        """Direct TP shard-to-shard redistribution: TP=4 → TP=2.

        Source: 4 ranks, each holding 1/4 of a [8,4] weight (shard_dim=0).
        Target: 2 ranks holding 1/2 each; ranks 2 and 3 become non-owners.

        Expected transfers (no gather step):
          src 0 → dst 0  (rows 0-1 → rows 0-1 of dst's [4,4] shard)
          src 1 → dst 0  (rows 2-3 → rows 2-3 of dst's [4,4] shard)
          src 2 → dst 1  (rows 4-5 → rows 0-1 of dst's [4,4] shard)
          src 3 → dst 1  (rows 6-7 → rows 2-3 of dst's [4,4] shard)
        """
        torch.manual_seed(42)
        full_weight = torch.randn(8, 4)  # full param, same on all ranks
        dist.broadcast(full_weight, src=0)

        # Each rank holds 1/4 of the rows.
        rows_per_rank = 2  # 8 / 4
        my_shard = full_weight[self.rank * rows_per_rank:(self.rank + 1) * rows_per_rank].clone()

        model = SimpleModel()
        model.param1.data = my_shard  # use param1 as the TP-sharded weight

        # --- Verify _get_transfer_pieces logic (pure, no communication) ---
        # Build source ownership: each rank owns a 2-row shard.
        src_ownership = {}
        for r in range(self.world_size):
            sr = (r * rows_per_rank, (r + 1) * rows_per_rank)
            src_ownership[r] = LayerOwnership(
                rank=r,
                layer_names=["param1"],
                is_placeholder={"param1": False},
                shard_range={"param1": sr},
                shard_dim={"param1": 0},
            )
        # Target ownership: ranks 0 and 1 each get 4 rows; ranks 2,3 are non-owners.
        tgt_ownership = {}
        for r in range(self.world_size):
            if r < 2:
                tr = (r * 4, (r + 1) * 4)
                tgt_ownership[r] = LayerOwnership(
                    rank=r,
                    layer_names=["param1"],
                    is_placeholder={"param1": False},
                    shard_range={"param1": tr},
                    shard_dim={"param1": 0},
                )
            else:
                tgt_ownership[r] = LayerOwnership(
                    rank=r,
                    layer_names=[],
                    is_placeholder={"param1": True},
                    shard_range={"param1": None},
                    shard_dim={"param1": None},
                )

        pieces = ReconfigurationExecutor._get_transfer_pieces(
            "param1", src_ownership, tgt_ownership
        )
        # Should be exactly 4 transfers (2 pieces per target rank).
        assert len(pieces) == 4, f"Expected 4 pieces, got {len(pieces)}: {pieces}"
        # All pieces send rows of size 2 (the source shard size).
        for p in pieces:
            assert p.src_local_end - p.src_local_start == 2, f"Wrong piece size: {p}"
            assert p.shard_dim == 0

        # --- Execute redistribution and verify assembled values ---
        executor = ReconfigurationExecutor(model)
        executor.execute(src_ownership, tgt_ownership)

        if self.rank < 2:
            expected = full_weight[self.rank * 4:(self.rank + 1) * 4]
            param = model.param1
            assert param is not None, f"Rank {self.rank} should own param1"
            assert param.shape == (4, 4), f"Rank {self.rank}: wrong shape {param.shape}"
            assert torch.allclose(param.data, expected, atol=1e-6), (
                f"Rank {self.rank}: assembled shard incorrect"
            )
        else:
            # Non-owner ranks should have their param replaced with placeholder.
            param = model.param1
            assert param is None or isinstance(param, type(None)), (
                f"Rank {self.rank} should not own param1, got {type(param)}"
            )

        print(f"Rank {self.rank}: TP shard-to-shard test passed!")
