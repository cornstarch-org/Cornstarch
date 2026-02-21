"""Unit tests for cornstarch.shardformer.layers.linear."""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

from cornstarch.shardformer.layers.linear import Linear1D_Row_ReduceScatter

from ...distributed_base import GlooDistributedTestBase

# Small but clearly non-trivial dimensions.
_SEQ = 4
_H_IN = 8
_H_OUT = 8


def _make_native_linear(seed: int = 42) -> nn.Linear:
    """Return a CPU nn.Linear with a deterministic weight (no bias)."""
    torch.manual_seed(seed)
    return nn.Linear(_H_IN, _H_OUT, bias=False)


class TestLinear1D_Row_ReduceScatter(GlooDistributedTestBase):
    """Tests for Linear1D_Row_ReduceScatter using a 2-process Gloo group."""

    @property
    def world_size(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _setup(self):
        """Build a sharded layer and helper tensors.

        Returns
        -------
        layer : Linear1D_Row_ReduceScatter  (weight shard on this rank)
        W_full : full weight tensor (H_OUT, H_IN), same on all ranks
        x_full : full input (SEQ, H_IN), same on all ranks
        x_split : this rank's input shard (SEQ, H_IN//tp)
        rank, tp : int
        """
        rank = dist.get_rank()
        tp = dist.get_world_size()

        native = _make_native_linear()
        W_full = native.weight.detach().clone()  # (H_OUT, H_IN)

        torch.manual_seed(7)
        x_full = torch.randn(_SEQ, _H_IN)
        x_split = x_full[:, rank * (_H_IN // tp) : (rank + 1) * (_H_IN // tp)].clone()

        layer = Linear1D_Row_ReduceScatter.from_native_module(
            native, dist.group.WORLD
        )
        return layer, W_full, x_full, x_split, rank, tp

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_weight_sharding(self):
        """from_native_module must shard the weight column-wise across ranks.

        ``shard_colwise`` splits along the last dimension of the weight
        matrix, which is the input-feature dimension for a row-parallel
        layer.  Rank j should hold ``W_full[:, j*H_IN//tp : (j+1)*H_IN//tp]``.
        """
        layer, W_full, _, _, rank, tp = self._setup()

        expected = W_full[:, rank * (_H_IN // tp) : (rank + 1) * (_H_IN // tp)]
        actual = layer.weight.data

        assert actual.shape == expected.shape, (
            f"rank {rank}: weight shape {actual.shape} != expected {expected.shape}"
        )
        torch.testing.assert_close(actual, expected, msg=f"weight mismatch on rank {rank}")

    def test_forward_output_shape_and_value(self):
        """Forward must produce the correct output shard.

        With ``parallel_input=True`` (default), rank j feeds
        ``x_full[:, j*H_IN//tp : (j+1)*H_IN//tp]``.  After the fused
        reduce-scatter the output shard on rank j must equal
        ``(x_full @ W_full.T)[:, j*H_OUT//tp : (j+1)*H_OUT//tp]``.
        """
        layer, W_full, x_full, x_split, rank, tp = self._setup()

        out = layer(x_split.clone())

        # Shape check
        expected_shape = (_SEQ, _H_OUT // tp)
        assert out.shape == expected_shape, (
            f"rank {rank}: output shape {out.shape} != {expected_shape}"
        )

        # Value check: manual reduce-scatter reference
        full_out = x_full @ W_full.T  # (SEQ, H_OUT)
        expected = full_out[:, rank * (_H_OUT // tp) : (rank + 1) * (_H_OUT // tp)]

        torch.testing.assert_close(
            out, expected, rtol=1e-4, atol=1e-5,
            msg=f"output value mismatch on rank {rank}"
        )

    def test_backward_gradient(self):
        """Backward must yield weight gradients matching a non-parallel reference.

        The reduce-scatter backward is an all-gather.  With loss = output.sum()
        every element of grad_output is 1, so the all-gather simply
        broadcasts ones to all ranks.  The weight gradient on rank j is then

            ones(H_OUT, SEQ) @ x_split_j  ==  ref_weight_grad[:, j*H_IN//tp:(j+1)*H_IN//tp]

        where ``ref_weight_grad = ones(H_OUT, SEQ) @ x_full``.
        """
        layer, W_full, x_full, x_split, rank, tp = self._setup()

        out = layer(x_split.clone())
        out.sum().backward()

        # Reference: non-distributed linear
        ref = nn.Linear(_H_IN, _H_OUT, bias=False)
        ref.weight = nn.Parameter(W_full.clone())
        ref(x_full.detach()).sum().backward()

        expected_grad = ref.weight.grad[
            :, rank * (_H_IN // tp) : (rank + 1) * (_H_IN // tp)
        ]

        torch.testing.assert_close(
            layer.weight.grad, expected_grad, rtol=1e-4, atol=1e-5,
            msg=f"weight gradient mismatch on rank {rank}"
        )


if __name__ == "__main__":
    run_tests()
