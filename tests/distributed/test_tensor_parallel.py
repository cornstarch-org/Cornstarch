"""Tests for DTensor-based tensor parallelism.

Uses a tiny 2-rank gloo setup.  Each test creates a small Linear layer,
shards it with ColwiseParallel / RowwiseParallel, runs a forward pass, and
checks that the output matches a reference computed on the full (unsharded)
weight.
"""
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel

from tests.distributed.distributed_base import GlooDistributedTestBase


class TestColwiseParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_colwise_forward_matches_reference(self):
        """Sharded forward should produce the same result as full forward."""
        tp_mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("tp",))
        rank = dist.get_rank()

        in_features, out_features = 8, 16
        torch.manual_seed(0)
        ref_linear = nn.Linear(in_features, out_features, bias=False)

        # Each rank gets a copy of the full weight to shard
        sharded_linear = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            sharded_linear.weight.copy_(ref_linear.weight)

        parallelize_module(sharded_linear, tp_mesh, ColwiseParallel())

        torch.manual_seed(42)
        x = torch.randn(4, in_features)

        with torch.no_grad():
            ref_out = ref_linear(x)  # (4, 16)
            sharded_out = sharded_linear(x)

        # ColwiseParallel with use_local_output=True (default) returns a plain
        # tensor with the local shard of shape (4, out_features // world_size).
        local_out = sharded_out.to_local() if hasattr(sharded_out, "to_local") else sharded_out
        shard_dim = out_features // self.world_size
        self.assertEqual(local_out.shape, (4, shard_dim))

        gathered = [torch.zeros(4, shard_dim) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out)
        full_out = torch.cat(gathered, dim=-1)  # (4, 16)

        self.assertTrue(
            torch.allclose(full_out, ref_out, atol=1e-5),
            f"Max diff: {(full_out - ref_out).abs().max():.6f}",
        )

    def test_rowwise_forward_matches_reference(self):
        """RowwiseParallel output should match full-weight matmul."""
        tp_mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("tp",))
        rank = dist.get_rank()

        in_features, out_features = 16, 8
        torch.manual_seed(0)
        ref_linear = nn.Linear(in_features, out_features, bias=False)

        sharded_linear = nn.Linear(in_features, out_features, bias=False)
        with torch.no_grad():
            sharded_linear.weight.copy_(ref_linear.weight)

        parallelize_module(sharded_linear, tp_mesh, RowwiseParallel())

        torch.manual_seed(42)
        x = torch.randn(4, in_features)

        with torch.no_grad():
            ref_out = ref_linear(x)  # (4, 8)

        # RowwiseParallel has input_layouts=Shard(-1), so it expects the
        # local shard as input (DTensor treats the passed tensor as the
        # local piece).  output_layouts=Replicate means the module does
        # the all-reduce internally.
        local_x = x.chunk(self.world_size, dim=-1)[rank]  # (4, 8)

        with torch.no_grad():
            sharded_out = sharded_linear(local_x)

        full_out = sharded_out.to_local() if hasattr(sharded_out, "to_local") else sharded_out

        self.assertTrue(
            torch.allclose(full_out, ref_out, atol=1e-5),
            f"Max diff: {(full_out - ref_out).abs().max():.6f}",
        )


if __name__ == "__main__":
    unittest.main()
