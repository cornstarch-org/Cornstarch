"""Distributed language-model pretraining via the Option C surface.

Run with torchrun, e.g. 8 GPUs as DP=2, PP=2, TP=2::

    torchrun --nproc_per_node=8 examples/distributed/pretrain_llm.py \
        --tp 2 --pp 2 --dp 2

Everything parallelism-specific is expressed declaratively: one ``ParallelConfig``
describes the LM's degrees, ``plan.distribute`` applies TP/PP (and EP for MoE)
and materializes, and the returned ``ctx`` folds the DP sampler into the
dataloader, builds the schedule, and exposes ``sync_gradients``.  There is no
rank math, no ordering rule, and no grad-sync wiring in this script.
"""
from __future__ import annotations

import tyro

import torch

from common import (
    DTYPE,
    FakeTextDataset,
    build_language_model,
    build_language_model_plan,
    causal_lm_criterion,
    init_distributed,
    train_loop,
)

from cornstarch.distributed import ParallelConfig, ParallelizationPlan


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

    model = build_language_model(model_name_or_path)
    vocab_size = model.hf_config.vocab_size

    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        model,
        ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            context_parallel_size=cp,
            expert_parallel_size=ep,
            data_parallel_size=dp,
        ),
    )
    ctx = plan.distribute(device, dtype=DTYPE)

    dataset = FakeTextDataset(vocab_size, seq_len)
    loader = ctx.prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)

    exec_plan, output_future = build_language_model_plan(model)
    schedule = ctx.create_schedule(
        exec_plan,
        output_future,
        num_microbatches=num_microbatches if pp > 1 else 1,
        microbatch_size=batch_size // num_microbatches if pp > 1 else 1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loop(ctx, schedule, loader, optimizer, causal_lm_criterion, steps)


if __name__ == "__main__":
    tyro.cli(main)
