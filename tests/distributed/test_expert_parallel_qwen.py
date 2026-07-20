"""Expert parallelism on a batched-expert MoE model (Qwen3.5 MoE).

Runs on 2 gloo ranks with ``num_experts=4`` so each rank owns 2 experts
(``num_experts % ep_size == 0`` with ``experts_per_rank > 1``).  Verifies:

- ``apply_expert_parallel`` slices the stacked expert tensors to this rank's
  shard and tags them ``_is_expert_parallel``;
- the EP forward reproduces the non-EP forward (every expert is computed
  exactly once, on its owner, and the results are gathered back);
- forward + backward produces finite gradients for both the sharded expert
  weights and the replicated (router / shared-expert / attention) weights.
"""
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import qwen3_5_moe_config

from cornstarch.distributed.data_parallel import GradientSynchronizer
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.models.conversions.qwen3_5_moe import _reduce_router_statistics
from cornstarch.models import from_hf_config
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
)


VOCAB_SIZE = 64


def _router_statistics(logits: torch.Tensor, top_k: int = 2) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1)
    selected = torch.topk(probabilities, top_k, dim=-1).indices
    counts = F.one_hot(selected, num_classes=logits.shape[-1]).float().sum(0)
    return torch.cat(
        [counts.reshape(-1), probabilities.sum(0), logits.new_tensor([logits.shape[0]])]
    )


def _router_aux(statistics: torch.Tensor, experts: int = 4, top_k: int = 2) -> torch.Tensor:
    count_size = experts * top_k
    counts = statistics[:count_size].reshape(top_k, experts)
    probability_sums = statistics[count_size : count_size + experts]
    tokens = statistics[-1]
    return experts * torch.sum((counts / tokens) * (probability_sums / tokens)[None])


def _router_reference(inputs: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference_weight = nn.Parameter(weight.detach().clone())
    loss = _router_aux(_router_statistics(inputs @ reference_weight))
    loss.backward()
    return loss.detach(), reference_weight.grad


def _make_moe_llm():
    config = qwen3_5_moe_config()
    config.vocab_size = VOCAB_SIZE
    config.tie_word_embeddings = False
    model = from_hf_config(
        config, model_kind="language", attn_implementation="eager"
    )
    model.set_random_init()
    model.materialize("cpu")
    model.train()
    return model


class TestQwenBatchedExpertParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_forward_matches_non_ep(self):
        """EP forward (2 experts/rank) equals the replicated non-EP forward."""
        ep_group = dist.group.WORLD
        ep_size = dist.get_world_size(ep_group)

        model = _make_moe_llm()
        num_experts = model.hf_config.num_experts
        experts_per_rank = num_experts // ep_size

        # Same input on every rank (seed is reset to 0 in the test harness),
        # so the per-rank EP output must match the full replicated reference.
        model.eval()
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        with torch.no_grad():
            reference = model(input_ids=input_ids).logits

        apply_expert_parallel(model, ep_group)

        # Each rank now holds only its slice of the stacked expert tensors.
        experts = model.decoder_layers[0].mlp.experts
        self.assertEqual(experts.gate_up_proj.shape[0], experts_per_rank)
        self.assertEqual(experts.down_proj.shape[0], experts_per_rank)
        self.assertTrue(getattr(experts.gate_up_proj, "_is_expert_parallel", False))
        self.assertTrue(getattr(experts.down_proj, "_is_expert_parallel", False))

        with torch.no_grad():
            ep_logits = model(input_ids=input_ids).logits

        self.assertTrue(
            torch.allclose(ep_logits, reference, atol=1e-4, rtol=1e-4),
            f"max abs diff {((ep_logits - reference).abs().max()).item()}",
        )

    def test_backward_grads(self):
        """Forward + backward yields finite grads for sharded and replicated params."""
        ep_group = dist.group.WORLD

        model = _make_moe_llm()
        apply_expert_parallel(model, ep_group)

        input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids=input_ids, labels=labels)
        self.assertTrue(out.loss.isfinite().all())
        out.loss.backward()

        experts = model.decoder_layers[0].mlp.experts
        self.assertIsNotNone(experts.gate_up_proj.grad)
        self.assertTrue(experts.gate_up_proj.grad.isfinite().all())

        # A replicated parameter (the router) must also receive a gradient.
        router_weight = model.decoder_layers[0].mlp.gate.weight
        self.assertIsNotNone(router_weight.grad)
        self.assertFalse(getattr(router_weight, "_is_expert_parallel", False))

    def test_gradient_sync_skips_experts(self):
        """DP sync averages replicated grads but leaves sharded experts alone."""
        ep_group = dist.group.WORLD

        model = _make_moe_llm()
        apply_expert_parallel(model, ep_group)

        # Different data per rank -> replicated-param grads differ before sync.
        torch.manual_seed(100 + dist.get_rank(ep_group))
        input_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
        labels = input_ids.clone()
        model(input_ids=input_ids, labels=labels).loss.backward()

        expert_grad = model.decoder_layers[0].mlp.experts.gate_up_proj.grad
        router_grad = model.decoder_layers[0].mlp.gate.weight.grad
        expert_before = expert_grad.clone()
        router_before = router_grad.clone()

        grad_sync = GradientSynchronizer(ep_group)
        grad_sync.register(model)
        grad_sync.sync()

        # Expert grads are per-rank and must be untouched by the sync.
        self.assertTrue(torch.equal(expert_grad, expert_before))

        # Router grad must change (it was averaged) and now match across ranks.
        self.assertFalse(torch.equal(router_grad, router_before))
        gathered = [torch.empty_like(router_grad) for _ in range(self.world_size)]
        dist.all_gather(gathered, router_grad.contiguous())
        for other in gathered:
            self.assertTrue(torch.allclose(other, router_grad, atol=1e-5))


class TestQwenRouterAuxDataParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_global_aux_loss_and_router_gradient_match_hf(self):
        """DP ranks reduce sufficient token statistics, not router activations."""
        config = qwen3_5_moe_config()
        config.vocab_size = VOCAB_SIZE
        config.router_aux_loss_coef = 0.03
        torch.manual_seed(19)
        hf_model = Qwen3_5MoeForCausalLM(config)
        model = from_hf_config(
            config, model_kind="language", attn_implementation="eager"
        )
        model.set_checkpoint_init(state_dict=hf_model.state_dict())
        model.materialize("cpu")
        model._dp_group = dist.group.WORLD

        generator = torch.Generator().manual_seed(991)
        full_ids = torch.randint(
            0, VOCAB_SIZE, (4, 8), generator=generator
        )
        local_ids = full_ids.chunk(self.world_size, dim=0)[dist.get_rank()]
        hf_output = hf_model(
            input_ids=full_ids, output_router_logits=True
        )
        output = model(
            input_ids=local_ids, output_router_logits=True
        )
        torch.testing.assert_close(
            output.aux_loss, hf_output.aux_loss, atol=1e-5, rtol=1e-5
        )

        output.aux_loss.backward()
        synchronizer = GradientSynchronizer(dist.group.WORLD)
        synchronizer.register(model)
        synchronizer.sync()
        hf_output.aux_loss.backward()
        torch.testing.assert_close(
            model.decoder_layers[0].mlp.gate.weight.grad,
            hf_model.model.layers[0].mlp.gate.weight.grad,
            atol=1e-5,
            rtol=1e-5,
        )


class TestQwenRouterAuxContextParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_cp_sum_sync_matches_global_router_gradient(self) -> None:
        generator = torch.Generator().manual_seed(2027)
        inputs = torch.randn(8, 3, generator=generator)
        initial_weight = torch.randn(3, 4, generator=generator)
        reference_loss, reference_grad = _router_reference(inputs, initial_weight)

        weight = nn.Parameter(initial_weight.clone())
        local_inputs = inputs.chunk(self.world_size)[dist.get_rank()]
        statistics = _reduce_router_statistics(
            _router_statistics(local_inputs @ weight),
            cp_group=dist.group.WORLD,
            dp_group=None,
        )
        loss = _router_aux(statistics)
        loss.backward()
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)

        torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weight.grad, reference_grad, atol=1e-6, rtol=1e-6)


class TestQwenRouterAuxContextAndDataParallel(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def test_cp_sum_then_dp_average_matches_global_router_gradient(self) -> None:
        rank = dist.get_rank()
        cp_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        cp_group = cp_groups[rank // 2]
        dp_group = dp_groups[rank % 2]

        generator = torch.Generator().manual_seed(2028)
        inputs = torch.randn(2, 8, 3, generator=generator)
        initial_weight = torch.randn(3, 4, generator=generator)
        reference_loss, reference_grad = _router_reference(
            inputs.flatten(0, 1), initial_weight
        )

        weight = nn.Parameter(initial_weight.clone())
        dp_index, cp_index = divmod(rank, 2)
        local_inputs = inputs[dp_index].chunk(2)[cp_index]
        statistics = _reduce_router_statistics(
            _router_statistics(local_inputs @ weight),
            cp_group=cp_group,
            dp_group=dp_group,
        )
        loss = _router_aux(statistics)
        loss.backward()
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM, group=dp_group)
        weight.grad.div_(2)

        torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(weight.grad, reference_grad, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
