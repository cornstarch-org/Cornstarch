"""Distributed language-model pretraining with a per-module parallel plan.

Run with torchrun, for example eight GPUs as DP=2, PP=2, TP=2::

    torchrun --nproc-per-node=8 --module examples.distributed.pretrain_llm \
        --tp 2 --pp 2 --dp 2

For multiple nodes, add the normal ``--nnodes``, ``--node-rank``,
``--master-addr``, and ``--master-port`` torchrun arguments. Each process binds
``LOCAL_RANK`` with ``torch.cuda.set_device`` before initializing NCCL.

This script reads top-to-bottom like the local examples. The schedule's
``step`` runs forward + criterion + backward (or the 1F1B microbatch loop under
PP), and ``ctx.sync_gradients`` all-reduces gradients across DP ranks.
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

from .common import (
    DTYPE,
    FakeTextDataset,
    causal_lm_criterion,
    context_parallel_language_inputs,
    init_distributed,
    local_trainable_parameters,
    microbatch_collate,
    move_microbatches_to_device,
)

from cornstarch.distributed import (
    ParallelConfig,
    ParallelContext,
    ParallelizationPlan,
    HeadTailContextParallelSplitter,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)


def _build_plan(language_model: Any, *, context_parallel: bool = False):
    """Build the (parallelism-agnostic) execution plan for an LM step."""
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
        language_model_inputs=context_parallel_language_inputs(context_parallel),
    )
    output_future = plan.run_language_model(module=language_model, inputs=merged)
    return plan, output_future


def _training_step(
    language_model: Any,
    ctx: ParallelContext,
    microbatches: list[dict[str, torch.Tensor]],
    criterion: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor],
) -> dict[str, Any]:
    """Run one optimizer step over the ``collate_fn`` microbatch list.

    ``CornstarchExecutionPlan`` is parallelism-agnostic, so the only thing that
    differs is how the microbatches are consumed:

    - **No pipeline parallelism** (modules co-located on every rank): iterate the
      microbatches, running the plan locally and accumulating gradients — exactly
      gradient accumulation, no schedule.
    - **Pipeline parallelism**: hand the microbatch list to the schedule, which
      drives the same futures DAG over a 1F1B program.

    ``ctx.sync_gradients`` all-reduces the DP gradients before the optimizer step.
    """
    plan, output_future = _build_plan(
        language_model,
        context_parallel=ctx.get_splitter(language_model) is not None,
    )

    if not ctx.uses_pipeline_parallel:
        loss_total = None
        for mb in microbatches:
            output = output_future.execute(inputs=mb)
            loss = criterion(output, mb) / len(microbatches)
            loss.backward()
            loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
        ctx.sync_gradients()
        return {"loss": loss_total}

    schedule = ctx.create_schedule(plan, output_future)
    result = schedule.step(microbatches, criterion, return_loss=True)
    ctx.sync_gradients()
    return result


def pretrain(
    model_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    tp: int = 1,
    pp: int | None = None,
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
            context_parallel_splitter=(
                HeadTailContextParallelSplitter() if cp > 1 else None
            ),
            expert_parallel_size=ep,
            data_parallel_size=dp,
        ),
    )
    ctx = plan.materialize(device, dtype=DTYPE)
    language_model.train()

    dataset = FakeTextDataset(language_model.hf_config.vocab_size, seq_len)
    # collate_fn returns the microbatch list for one optimizer step.
    dataloader = ctx.prepare_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=microbatch_collate(num_microbatches),
        shuffle=True,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(local_trainable_parameters(language_model), lr=lr)
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
            microbatches = move_microbatches_to_device(next(dataloader_iter), device)
            outputs = _training_step(
                language_model,
                ctx,
                microbatches,
                causal_lm_criterion,
            )
            loss = outputs["loss"]
            if loss is not None:
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
