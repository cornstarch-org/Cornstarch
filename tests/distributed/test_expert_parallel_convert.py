"""ModuleList -> batched conversion + expert parallelism.

No supported model still stores experts as an ``nn.ModuleList`` (modern HF MoEs
are all batched), so this exercises the conversion path on a synthetic
ModuleList MoE block:

- ``_convert_modulelist_block_to_batched`` reproduces the block's output after
  stacking the per-expert weights into the batched representation;
- ``_parallelize_experts`` then shards those stacked tensors across 2 gloo ranks
  (4 experts -> 2 per rank) and the EP forward matches the non-EP batched output;
- backward produces finite grads for the sharded experts and the (replicated)
  gate.
"""
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase

from cornstarch.distributed.expert_parallel import (
    _convert_modulelist_block_to_batched,
    _is_batched_experts,
    _parallelize_experts,
)
from cornstarch.distributed.expert_parallel.routing import ExpertRouter


HIDDEN = 16
INTERMEDIATE = 32
NUM_EXPERTS = 4
TOP_K = 2


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _ModuleListMoE(nn.Module):
    """Top-k MoE block whose experts live in an nn.ModuleList."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([_Expert() for _ in range(NUM_EXPERTS)])
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        # Match the routing the converter uses so conversion is value-preserving.
        self._router = ExpertRouter(NUM_EXPERTS, TOP_K)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        weights, idx = self._router(self.gate(x))
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            for k in range(self.top_k):
                mask = idx[:, k] == e
                if mask.any():
                    contribution = weights[mask, k, None] * self.experts[e](x[mask])
                    out = out.index_add(
                        0, mask.nonzero(as_tuple=True)[0], contribution
                    )
        return out.reshape(shape)


class TestModuleListConversionEP(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_convert_then_parallelize(self):
        ep_group = dist.group.WORLD
        ep_size = dist.get_world_size(ep_group)

        block = _ModuleListMoE()
        for p in block.parameters():
            with torch.no_grad():
                p.normal_(0.0, 0.1)

        x = torch.randn(2, 6, HIDDEN)
        with torch.no_grad():
            modulelist_out = block(x)

        # Conversion preserves the block output.
        _convert_modulelist_block_to_batched(block)
        self.assertTrue(_is_batched_experts(block.experts))
        with torch.no_grad():
            batched_out = block(x)
        self.assertTrue(
            torch.allclose(batched_out, modulelist_out, atol=1e-5),
            f"conversion diff {((batched_out - modulelist_out).abs().max()).item()}",
        )

        # Expert parallelism over the converted batched experts.
        _parallelize_experts(
            block.experts, dist.get_rank(ep_group), ep_size, ep_group
        )
        self.assertEqual(block.experts.gate_up_proj.shape[0], NUM_EXPERTS // ep_size)
        self.assertTrue(
            getattr(block.experts.gate_up_proj, "_is_expert_parallel", False)
        )

        with torch.no_grad():
            ep_out = block(x)
        self.assertTrue(
            torch.allclose(ep_out, batched_out, atol=1e-5),
            f"EP diff {((ep_out - batched_out).abs().max()).item()}",
        )

        # Backward reaches the sharded experts and the replicated gate.
        block(x).sum().backward()
        self.assertIsNotNone(block.experts.gate_up_proj.grad)
        self.assertTrue(block.experts.gate_up_proj.grad.isfinite().all())
        self.assertIsNotNone(block.gate.weight.grad)


if __name__ == "__main__":
    unittest.main()
