"""Distributed language-model pretraining via the Option C surface.

Run with torchrun, e.g. 8 GPUs as DP=2, PP=2, TP=2::

    torchrun --nproc_per_node=8 examples/distributed/pretrain_llm.py \
        --tp 2 --pp 2 --dp 2

This script reads top-to-bottom like the non-distributed
``examples/pretrain_vlm.py`` — the training loop is intentionally identical.
The only distributed-specific calls are pushed into ``_training_step``: the
schedule's ``step`` runs forward + criterion + backward (or the 1F1B microbatch
loop under PP) and ``ctx.sync_gradients`` all-reduces gradients across DP ranks.
Building the model differs only in ``plan.parallelize`` + ``plan.materialize``
(instead of ``model.materialize``) and ``ctx.prepare_dataloader`` (DP sampler +
CP split folded in).  There is no rank math, no ordering rule, and no grad-sync
wiring here.
"""
from __future__ import annotations

from typing import Any, Callable

import tyro

import torch
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

from common import (
    DTYPE,
    FakeTextDataset,
    causal_lm_criterion,
    init_distributed,
)

from cornstarch.distributed import (
    ParallelConfig,
    ParallelContext,
    ParallelizationPlan,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)


def _training_step(
    language_model: Any,
    ctx: ParallelContext,
    batch: dict[str, torch.Tensor],
    criterion: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_microbatches: int,
    microbatch_size: int,
) -> dict[str, Any]:
    """Build the plan for this batch, run one schedule step, and sync gradients.

    Just like the non-distributed ``examples/pretrain_vlm.py``, the execution
    plan is rebuilt **every step** — schedule construction is cheap (DAG analysis
    plus a rank-local program, no collectives).  ``schedule.step`` runs forward +
    criterion + backward (or the 1F1B microbatch loop under PP); then
    ``ctx.sync_gradients`` all-reduces the DP gradients before the optimizer step.
    Returns the schedule result whose ``"loss"`` is the step loss (``None`` on
    non-last pipeline stages).
    """
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
    )
    output_future = plan.run_language_model(module=language_model, inputs=merged)

    schedule = ctx.create_schedule(
        plan,
        output_future,
        num_microbatches=num_microbatches,
        microbatch_size=microbatch_size,
    )
    result = schedule.step(batch, criterion, optimizer, return_loss=True)
    ctx.sync_gradients()
    return result


def pretrain(
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
    dataloader = ctx.prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(language_model.parameters(), lr=lr)
    optimizer.zero_grad()

    num_warmup_steps = int(steps * 0.1)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=steps,
    )

    dataloader_iter = iter(dataloader)
    with tqdm(range(steps), disable=rank != 0) as pbar:
        for _ in pbar:
            batch = next(dataloader_iter)
            outputs = _training_step(
                language_model,
                ctx,
                batch,
                causal_lm_criterion,
                optimizer,
                num_microbatches=num_microbatches if pp > 1 else 1,
                microbatch_size=batch_size // num_microbatches if pp > 1 else 1,
            )
            loss = outputs["loss"]
            if loss is not None:
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
