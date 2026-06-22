"""Distributed VLM pretraining via the Option C per-modality surface.

Run with torchrun, e.g. 4 GPUs with the vision encoder tensor-parallel and the
language model tensor + pipeline parallel, all replicated once (dp=1)::

    torchrun --nproc_per_node=4 examples/distributed/pretrain_vlm.py \
        --vision-tp 2 --llm-tp 2 --llm-pp 1

This reads top-to-bottom like the non-distributed ``examples/pretrain_vlm.py`` —
the training loop is intentionally identical.  The only distributed-specific
calls are pushed into ``_training_step`` (the schedule's ``step`` does forward +
criterion + backward, and ``ctx.sync_gradients`` all-reduces DP gradients).
Building the models differs only in the per-modality ``plan.parallelize`` +
``plan.materialize`` and ``ctx.prepare_dataloader``.

Each modality is described independently: ``plan.parallelize`` is called once
per modality with its own ``ParallelConfig`` (vision can be TP-only while the
LLM is TP+PP), and ``plan.materialize`` resolves the per-modality + DP-offset
rank math and materializes both models.  The projector follows its encoder
automatically — there is no standalone projector setup in this script.
"""
from __future__ import annotations

from typing import Any, Callable

import tyro

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

from common import DTYPE, causal_lm_criterion, init_distributed, microbatch_collate

from cornstarch.distributed import (
    ParallelConfig,
    ParallelContext,
    ParallelizationPlan,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    build_modality_encoder,
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
        pixel_values = torch.randn(3, *self.image_size, generator=g)
        return {"input_ids": ids, "labels": ids.clone(), "pixel_values": pixel_values}


def _build_plan(language_model: Any, modality_encoder: Any, mb: dict):
    """Build the (parallelism-agnostic) plan for one VLM microbatch.

    The three plan-construction lines match the non-distributed
    ``examples/pretrain_vlm.py``; ``mb`` is one microbatch from the user's
    ``collate_fn`` list.
    """
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=modality_encoder,
        pixel_values=mb["pixel_values"],
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=mb["input_ids"],
        labels=mb["labels"],
        modality_token_ids={"vision": IMAGE_TOKEN_ID},
        encoder_outputs={"vision": vision_outputs},
    )
    output_future = plan.run_language_model(module=language_model, inputs=merged)
    return plan, output_future


def _training_step(
    language_model: Any,
    modality_encoder: Any,
    ctx: ParallelContext,
    microbatches: list[dict[str, torch.Tensor]],
    criterion: Callable[[Any, dict[str, torch.Tensor]], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Run one optimizer step over the ``collate_fn`` microbatch list.

    Without pipeline parallelism the encoder and language model are co-located on
    every rank, so the whole ``encoder -> merge -> language model`` plan runs
    locally per microbatch and gradients accumulate (no schedule). With pipeline
    parallelism the encoder and language model are disaggregated onto separate
    ranks and the microbatch list is driven by the schedule (wired in T008).
    ``ctx.sync_gradients`` all-reduces the DP gradients before the optimizer step.
    """
    if not ctx.uses_pipeline_parallel:
        loss_total = None
        for mb in microbatches:
            _, output_future = _build_plan(language_model, modality_encoder, mb)
            output = output_future.execute()
            loss = criterion(output, mb) / len(microbatches)
            loss.backward()
            loss_total = loss.detach() if loss_total is None else loss_total + loss.detach()
        ctx.sync_gradients()
        return {"loss": loss_total}

    plan, output_future = _build_plan(language_model, modality_encoder, microbatches[0])
    schedule = ctx.create_schedule(plan, output_future)
    result = schedule.step(microbatches, criterion, optimizer, return_loss=True)
    ctx.sync_gradients()
    return result


def pretrain(
    vision_name_or_path: str = "openai/clip-vit-base-patch32",
    llm_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    vision_tp: int = 1,
    llm_tp: int = 1,
    llm_pp: int | None = None,
    dp: int = 1,
    batch_size: int = 4,
    seq_len: int = 128,
    num_microbatches: int = 2,
    steps: int = 10,
    lr: float = 1e-4,
) -> None:
    rank, world_size, device = init_distributed()

    vision_config = getattr(
        AutoConfig.from_pretrained(vision_name_or_path), "vision_config",
        AutoConfig.from_pretrained(vision_name_or_path),
    )
    llm_config = AutoConfig.from_pretrained(llm_name_or_path)

    vision_encoder = from_hf_config(vision_config, model_kind="vision")
    language_model = from_hf_config(llm_config, model_kind="language")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )

    language_model.set_random_init()
    modality_encoder.set_random_init()

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
            tensor_parallel_size=vision_tp,
            pipeline_parallel_size=vision_pp,
            data_parallel_size=dp,
        ),
    )
    plan.parallelize(
        language_model,
        ParallelConfig(
            tensor_parallel_size=llm_tp,
            pipeline_parallel_size=llm_pp,
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
    )

    params = list(language_model.parameters()) + list(modality_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
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
            microbatches = next(dataloader_iter)
            outputs = _training_step(
                language_model,
                modality_encoder,
                ctx,
                microbatches,
                causal_lm_criterion,
                optimizer,
            )
            loss = outputs["loss"]
            if loss is not None:
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
