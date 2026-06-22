"""End-to-end integration tests for the Option C ``ParallelizationPlan`` surface.

Each combination builds a tiny model, drives it through the *declarative*
surface (``ParallelizationPlan.parallelize`` -> ``.materialize`` ->
``ParallelContext.create_schedule`` / ``.sync_gradients``), runs one
forward+backward step, and asserts the loss is finite and gradients exist.

Two matrices are exercised:

- ``TestLanguageModelComposition`` — a dense Llama over curated DP/PP/TP
  combinations (the model-runnable-on-CPU dimensions), validating that those
  parallelisms compose through the real surface.
- ``TestExpertParallelComposition`` — a Qwen3.5-MoE model where expert
  parallelism (a real mesh axis) composes with DP/PP/TP.

Context parallelism is exercised by its splitter tests and the guarded
numerical-equivalence test (its attention kernel is CUDA-only), so ``cp`` stays
1 here.  All combinations keep ``world_size <= 8`` to stay cheap under gloo.
"""
from __future__ import annotations

import re
import unittest

import torch
import torch.distributed as dist

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import llama_config

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)


VOCAB = 128


def _world_size_from_name(name: str) -> int:
    m = re.search(r"dp(\d+)_pp(\d+)_cp(\d+)_tp(\d+)_ep(\d+)", name)
    assert m, f"cannot parse parallel dims from {name}"
    dp, pp, cp, tp, ep = map(int, m.groups())
    return dp * pp * cp * tp * ep


def _name(combo: tuple[int, int, int, int, int]) -> str:
    dp, pp, cp, tp, ep = combo
    return f"dp{dp}_pp{pp}_cp{cp}_tp{tp}_ep{ep}"


def _moe_config() -> Qwen3_5MoeTextConfig:
    """A tiny Qwen3.5-MoE config with only full-attention layers.

    The gated-delta-net *linear*-attention layer produces NaNs when isolated as
    a pipeline boundary stage (a pre-existing model/PP interaction, tracked in
    tasks/backlog.md). Using full-attention layers keeps EP-with-PP composition
    testable; linear-attention EP (without PP) is covered by
    ``test_expert_parallel_qwen``.
    """
    return Qwen3_5MoeTextConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "full_attention"],
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        tie_word_embeddings=False,
    )


def _build_model(moe: bool) -> object:
    config = _moe_config() if moe else llama_config()
    config.vocab_size = VOCAB
    config.tie_word_embeddings = False
    model = from_hf_config(config, model_kind="language", attn_implementation="eager")
    model.set_random_init()
    return model


def _run_combo(test: GlooDistributedTestBase, combo, moe: bool) -> None:
    dp, pp, cp, tp, ep = combo
    world_size = dp * pp * cp * tp * ep

    model = _build_model(moe)
    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        model,
        ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            context_parallel_size=cp,
            data_parallel_size=dp,
            expert_parallel_size=ep,
        ),
    )
    ctx = plan.materialize("cpu", dtype=torch.float32)

    batch_size = max(4, pp * 2)
    torch.manual_seed(100 + dist.get_rank())
    batch = {
        "input_ids": torch.randint(0, VOCAB, (batch_size, 16)),
        "labels": torch.randint(0, VOCAB, (batch_size, 16)),
    }

    exec_plan = CornstarchExecutionPlan()
    merged = exec_plan.merge_modality_encoder_outputs(
        language_model=model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
    )
    output_future = exec_plan.run_language_model(module=model, inputs=merged)

    if pp > 1:
        num_microbatches = 2
        schedule = ctx.create_schedule(
            exec_plan, output_future,
            num_microbatches=num_microbatches,
            microbatch_size=batch_size // num_microbatches,
        )
    else:
        schedule = ctx.create_schedule(exec_plan, output_future)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def criterion(output, micro_batch):
        if isinstance(output, torch.Tensor):
            return output
        return output.loss if hasattr(output, "loss") else output["loss"]

    result = schedule.step(batch, criterion, optimizer, return_loss=True)
    if result["loss"] is not None:
        test.assertTrue(result["loss"].isfinite().all())

    ctx.sync_gradients()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    test.assertGreater(len(grads), 0, "no parameter received a gradient")


# DP / PP / TP compositions on a dense LM (cp=1, ep=1).
_LM_COMBOS = [
    (2, 1, 1, 1, 1),  # DP
    (1, 2, 1, 1, 1),  # PP
    (1, 1, 1, 2, 1),  # TP
    (2, 2, 1, 1, 1),  # DP + PP
    (1, 2, 1, 2, 1),  # PP + TP
    (2, 1, 1, 2, 1),  # DP + TP
    (2, 2, 1, 2, 1),  # DP + PP + TP
]

# EP-as-a-mesh-axis compositions on a MoE LM.
_EP_COMBOS = [
    (1, 1, 1, 1, 2),  # EP alone
    (2, 1, 1, 1, 2),  # EP + DP
    (1, 2, 1, 1, 2),  # EP + PP
    (1, 1, 1, 2, 2),  # EP + TP
    (1, 2, 1, 2, 2),  # EP + PP + TP
]


@instantiate_parametrized_tests
class TestLanguageModelComposition(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return _world_size_from_name(self._testMethodName)

    @parametrize("combo", _LM_COMBOS, name_fn=_name)
    def test(self, combo):
        _run_combo(self, combo, moe=False)


@instantiate_parametrized_tests
class TestExpertParallelComposition(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return _world_size_from_name(self._testMethodName)

    @parametrize("combo", _EP_COMBOS, name_fn=_name)
    def test(self, combo):
        _run_combo(self, combo, moe=True)


if __name__ == "__main__":
    unittest.main()
