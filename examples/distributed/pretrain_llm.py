"""Distributed language-model pretraining via the Option C surface.

Run with torchrun, e.g. 8 GPUs as DP=2, PP=2, TP=2::

    torchrun --nproc_per_node=8 examples/distributed/pretrain_llm.py \
        --tp 2 --pp 2 --dp 2

This script reads top-to-bottom like the non-distributed
``examples/pretrain_vlm.py``: build the model, set its init plan, build the
execution plan, and run an explicit training loop.  Only five lines are
distributed-specific: ``init_distributed()``, ``plan.parallelize`` +
``plan.materialize`` (instead of ``model.materialize``),
``ctx.prepare_dataloader`` (DP sampler + CP split folded in),
``ctx.create_schedule`` + ``schedule.step`` (instead of inline
execute/backward), and ``ctx.sync_gradients`` before the optimizer step.  There
is no rank math, no ordering rule, and no grad-sync wiring here.
"""
from __future__ import annotations

import tyro

import torch
from transformers import AutoConfig

from common import (
    DTYPE,
    FakeTextDataset,
    causal_lm_criterion,
    init_distributed,
)

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)


def main(
    model_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    ep: int = 1,
    dp: int = 1,
    batch_size: int = 8,
    seq_len: int = 128,
    num_microbatches: int = 2,
    steps: int = 10,
    lr: float = 1e-4,
) -> None:
    rank, world_size, device = init_distributed()

    config = AutoConfig.from_pretrained(model_name_or_path)
    config = getattr(config, "text_config", config)

    language_model = from_hf_config(config, model_kind="language")
    language_model.set_random_init()

    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        language_model,
        ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            context_parallel_size=cp,
            expert_parallel_size=ep,
            data_parallel_size=dp,
        ),
    )
    ctx = plan.materialize(device, dtype=DTYPE)
    language_model.train()

    dataset = FakeTextDataset(language_model.hf_config.vocab_size, seq_len)
    loader = ctx.prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)

    def build_plan():
        exec_plan = CornstarchExecutionPlan()
        merged = exec_plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={},
            encoder_outputs={},
        )
        output_future = exec_plan.run_language_model(
            module=language_model, inputs=merged
        )
        return exec_plan, output_future

    exec_plan, output_future = build_plan()
    schedule = ctx.create_schedule(
        exec_plan,
        output_future,
        num_microbatches=num_microbatches if pp > 1 else 1,
        microbatch_size=batch_size // num_microbatches if pp > 1 else 1,
    )

    optimizer = torch.optim.Adam(language_model.parameters(), lr=lr)
    optimizer.zero_grad()

    step = 0
    for batch in loader:
        if step >= steps:
            break
        result = schedule.step(batch, causal_lm_criterion, optimizer, return_loss=True)
        ctx.sync_gradients()
        optimizer.step()
        optimizer.zero_grad()
        if result["loss"] is not None and rank == 0:
            print(f"step {step}: loss {result['loss'].item():.4f}")
        step += 1


if __name__ == "__main__":
    tyro.cli(main)
