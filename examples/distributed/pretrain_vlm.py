"""Distributed VLM pretraining with distinct encoder and LLM grids.

This three-GPU example assigns one rank to the vision stage and a two-way TP
grid to the language-model stage::

    torchrun --nproc-per-node=3 --module examples.distributed.pretrain_vlm \
        --llm-tp 2 --llm-pp 1

The same command scales across nodes with torchrun's rendezvous arguments.
Every process binds its node-local GPU before NCCL initialization.

This reads top-to-bottom like the non-distributed ``examples/pretrain_vlm.py`` —
the training loop is intentionally identical.  The only distributed-specific
calls are pushed into ``_training_step`` (the schedule's ``step`` does forward +
criterion + backward, and ``ctx.sync_gradients`` all-reduces DP gradients).
Building the models differs only in the per-modality ``plan.parallelize`` +
``plan.materialize`` and ``ctx.prepare_dataloader``.

Each module is described independently: the vision module owns its pipeline
stage, while the LLM may use TP, CP, and PP. ``plan.materialize`` verifies that
their grids consume the whole world and materializes only local ownership. The
projector follows its encoder automatically.
"""
from __future__ import annotations

from typing import Any, Callable

import tyro

import torch
from peft import LoraConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

from .common import (
    DTYPE,
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
    FinetuningMode,
    build_modality_encoder,
    configure_finetuning,
    from_hf_config,
)


IMAGE_TOKEN_ID = 0


class FakeVLMDataset(Dataset):
    """Synthetic VLM batch: one image plus a caption with image placeholders."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_image_tokens: int,
        image_size: tuple[int, int],
        length: int = 4096,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_image_tokens = num_image_tokens
        self.image_size = image_size
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(index)
        ids = torch.randint(1, self.vocab_size, (self.seq_len,), generator=g)
        ids[: self.num_image_tokens] = IMAGE_TOKEN_ID  # image placeholder tokens
        labels = ids.clone()
        labels[: self.num_image_tokens] = -100
        pixel_values = torch.randn(3, *self.image_size, generator=g)
        return {"input_ids": ids, "labels": labels, "pixel_values": pixel_values}


def _build_plan(
    language_model: Any,
    modality_encoder: Any,
    *,
    context_parallel: bool = False,
):
    """Build the (parallelism-agnostic) VLM plan.

    The three plan-construction lines match the non-distributed
    ``examples/pretrain_vlm.py``; inputs are futures so the same plan serves both
    the local per-microbatch run and the schedule (which feeds each microbatch).
    """
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=modality_encoder,
        pixel_values=ExecutionFuture("pixel_values"),
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={"vision": IMAGE_TOKEN_ID},
        encoder_outputs={"vision": vision_outputs},
        language_model_inputs=context_parallel_language_inputs(context_parallel),
    )
    output_future = plan.run_language_model(module=language_model, inputs=merged)
    return plan, output_future


def _training_step(
    language_model: Any,
    modality_encoder: Any,
    ctx: ParallelContext,
    microbatches: list[dict[str, torch.Tensor]],
    criterion: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor],
) -> dict[str, Any]:
    """Run one optimizer step over the ``collate_fn`` microbatch list.

    Without pipeline parallelism the encoder and language model are co-located on
    every rank, so the whole ``encoder -> merge -> language model`` plan runs
    locally per microbatch and gradients accumulate (no schedule). With pipeline
    parallelism the encoder is the leading pipeline stage feeding the
    language-model stages, and the schedule drives the microbatch list across the
    cross-mesh seam. ``ctx.sync_gradients`` all-reduces the DP gradients before
    the optimizer step.
    """
    plan, output_future = _build_plan(
        language_model,
        modality_encoder,
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


def _lora_config(mode: FinetuningMode) -> LoraConfig | None:
    if mode != "lora":
        return None
    return LoraConfig(
        target_modules="all-linear",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )


def pretrain(
    vision_name_or_path: str = "openai/clip-vit-base-patch32",
    llm_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    llm_tp: int = 1,
    llm_pp: int | None = None,
    llm_cp: int = 1,
    dp: int = 1,
    vision_train_mode: FinetuningMode = "full",
    llm_train_mode: FinetuningMode = "full",
    batch_size: int = 4,
    seq_len: int = 128,
    num_microbatches: int = 2,
    steps: int = 10,
    lr: float = 1e-4,
) -> None:
    rank, world_size, device = init_distributed()

    vision_root_config = AutoConfig.from_pretrained(vision_name_or_path)
    vision_config = getattr(vision_root_config, "vision_config", vision_root_config)
    llm_root_config = AutoConfig.from_pretrained(llm_name_or_path)
    llm_config = getattr(llm_root_config, "text_config", llm_root_config)

    vision_encoder = from_hf_config(vision_config, model_kind="vision")
    language_model = from_hf_config(llm_config, model_kind="language")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )

    language_model.set_random_init()
    modality_encoder.set_random_init()
    configure_finetuning(
        modality_encoder,
        vision_train_mode,
        lora_config=_lora_config(vision_train_mode),
    )
    configure_finetuning(
        language_model,
        llm_train_mode,
        lora_config=_lora_config(llm_train_mode),
    )

    # Per-modality declarative configs. All modules must agree on pipeline
    # parallelism: when ``llm_pp`` is None there is no PP and the encoder + LLM are
    # co-located on every rank (each rank runs the whole model); when ``llm_pp`` is
    # a positive int the encoder becomes its own leading pipeline stage
    # (``pipeline_parallel_size=1``) disaggregated from the pipelined LLM.
    vision_pp = 1 if llm_pp is not None else None
    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        modality_encoder,
        ParallelConfig(
            pipeline_parallel_size=vision_pp,
            data_parallel_size=dp,
        ),
    )
    plan.parallelize(
        language_model,
        ParallelConfig(
            tensor_parallel_size=llm_tp,
            pipeline_parallel_size=llm_pp,
            context_parallel_size=llm_cp,
            context_parallel_splitter=(
                HeadTailContextParallelSplitter() if llm_cp > 1 else None
            ),
            data_parallel_size=dp,
        ),
    )
    ctx = plan.materialize(device, dtype=DTYPE)
    language_model.train()
    modality_encoder.train()

    patches_per_side = int(vision_config.image_size) // int(vision_config.patch_size)
    num_image_tokens = patches_per_side * patches_per_side + 1  # + CLS token
    image_size = (int(vision_config.image_size), int(vision_config.image_size))
    dataset = FakeVLMDataset(
        language_model.hf_config.vocab_size, seq_len, num_image_tokens, image_size
    )
    dataloader = ctx.prepare_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=microbatch_collate(num_microbatches),
        shuffle=True,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        local_trainable_parameters(language_model, modality_encoder), lr=lr
    )
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
                modality_encoder,
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
