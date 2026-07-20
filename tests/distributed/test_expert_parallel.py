"""Tests for expert-parallel dispatcher.

Uses a 2-rank gloo setup.  Each rank acts as 1 EP rank.  The test creates
a toy 2-expert MoE where each expert is a trivial identity (scale) transform,
then verifies that all tokens are processed exactly once and results are
numerically consistent with single-rank reference execution.
"""
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase
from cornstarch.distributed.expert_parallel.routing import (
    ExpertParallelDispatcher,
    ExpertRouter,
)


class ScaleExpert(nn.Module):
    """Trivial expert: multiplies input by a fixed scalar."""

    def __init__(self, scale: float):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class TestExpertRouter(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_router_output_shapes(self):
        """Router returns (N, top_k) weights and IDs."""
        router = ExpertRouter(num_experts=4, top_k=2)
        gate_logits = torch.randn(8, 4)
        weights, ids = router(gate_logits)
        self.assertEqual(weights.shape, (8, 2))
        self.assertEqual(ids.shape, (8, 2))

    def test_routing_weights_sum_to_one(self):
        """Softmax routing weights should sum to 1 per token."""
        router = ExpertRouter(num_experts=4, top_k=2)
        gate_logits = torch.randn(16, 4)
        weights, _ = router(gate_logits)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(16), atol=1e-5))


class TestExpertParallelDispatcher(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _get_ep_group(self) -> dist.ProcessGroup:
        return dist.group.WORLD

    def test_dispatch_collect_roundtrip(self):
        """Dispatched tokens, processed locally, then collected should match reference."""
        ep_group = self._get_ep_group()
        ep_rank = dist.get_rank(ep_group)
        ep_size = dist.get_world_size(ep_group)

        # 2 experts total, 1 per rank.
        experts = [ScaleExpert(scale=1.0), ScaleExpert(scale=2.0)]
        local_expert = experts[ep_rank]
        local_expert_start = ep_rank

        dispatcher = ExpertParallelDispatcher()
        router = ExpertRouter(num_experts=2, top_k=1)

        torch.manual_seed(0)
        N, d = 8, 4
        tokens = torch.randn(N, d)

        # Compute routing with fixed gate logits so rank 0 → expert 0, rank 1 → expert 1.
        # Use deterministic gate logits: odd tokens prefer expert 1, even prefer expert 0.
        gate_logits = torch.zeros(N, 2)
        for i in range(N):
            gate_logits[i, i % 2] = 10.0   # strong preference to alternate experts
        routing_weights, expert_ids = router(gate_logits)

        # Dispatch.
        local_tokens, local_expert_ids, recv_counts = dispatcher.dispatch(
            tokens, expert_ids, ep_group, num_experts=2
        )

        # Run local expert on received tokens.
        M = local_tokens.shape[0]
        expert_out = torch.zeros_like(local_tokens)
        if M > 0:
            expert_out = local_expert(local_tokens)

        # Collect.
        output = dispatcher.collect(expert_out, routing_weights, expert_ids, recv_counts, ep_group)

        # Verify shape.
        self.assertEqual(output.shape, (N, d))

        # Verify: each token's output should equal tokens[i] * scale[expert_ids[i, 0]].
        if ep_rank == 0:
            for i in range(N):
                eid = expert_ids[i, 0].item()
                expected = tokens[i] * experts[eid].scale
                self.assertTrue(
                    torch.allclose(output[i], expected, atol=1e-5),
                    f"Token {i}: expected {expected}, got {output[i]}",
                )


if __name__ == "__main__":
    unittest.main()
