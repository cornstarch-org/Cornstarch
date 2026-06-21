"""Distributed VLM pretraining via the Option C per-modality surface.

Run with torchrun, e.g. 4 GPUs with the vision encoder tensor-parallel and the
language model tensor + pipeline parallel, all replicated once (dp=1)::

    torchrun --nproc_per_node=4 examples/distributed/pretrain_vlm.py \
        --vision-tp 2 --llm-tp 2 --llm-pp 1

The point of Option C for multimodal models: each modality is described
independently.  ``plan.parallelize`` is called once per modality with its own
``ParallelConfig`` (vision can be TP-only while the LLM is TP+PP), and
``plan.distribute`` resolves the per-modality + DP-offset rank math, builds each
modality's mesh, applies the model-side parallelisms, and materializes — no
hand-written rank arithmetic in this script.

This example parallelizes the inner ``CornstarchVisionEncoder`` and
``CornstarchLanguageModel`` (both ``CornstarchModelBase``); the projector that
bridges them is then built and materialized around the parallelized encoder.
"""
from __future__ import annotations

import tyro

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import AutoConfig

from common import DTYPE, causal_lm_criterion, init_distributed, train_loop

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    CornstarchModalityEncoder,
    ExecutionFuture,
    from_hf_config,
)


IMAGE_TOKEN_ID = 0


class FakeVLMDataset(Dataset):
    """Synthetic VLM batch: one image plus a caption with image placeholders."""

    def __init__(self, vocab_size, seq_len, num_image_tokens, image_size, length=4096):
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


def main(
    vision_name_or_path: str = "openai/clip-vit-base-patch32",
    llm_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    vision_tp: int = 1,
    llm_tp: int = 1,
    llm_pp: int = 1,
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
    vision_encoder.set_random_init()
    language_model.set_random_init()

    # Per-modality declarative configs — vision and the LLM are parallelized
    # independently; distribute() handles the cross-modality rank assignment.
    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        vision_encoder,
        ParallelConfig(tensor_parallel_size=vision_tp, data_parallel_size=dp),
    )
    plan.parallelize(
        language_model,
        ParallelConfig(
            tensor_parallel_size=llm_tp,
            pipeline_parallel_size=llm_pp,
            data_parallel_size=dp,
        ),
    )
    ctx = plan.distribute(device, dtype=DTYPE)

    # Bridge the (already parallelized + materialized) vision encoder to the LLM
    # with a projector, then materialize the projector to match.
    modality_encoder = CornstarchModalityEncoder.from_encoder_and_language_model(
        vision_encoder, language_model, modality="vision", projector_type="linear"
    )
    modality_encoder.projector.set_random_init()
    modality_encoder.projector.materialize(device).to(dtype=DTYPE)

    patches_per_side = int(vision_config.image_size) // int(vision_config.patch_size)
    num_image_tokens = patches_per_side * patches_per_side + 1  # + CLS token
    image_size = (int(vision_config.image_size), int(vision_config.image_size))
    dataset = FakeVLMDataset(
        language_model.hf_config.vocab_size, seq_len, num_image_tokens, image_size
    )
    loader = ctx.prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)

    def build_plan():
        plan_exec = CornstarchExecutionPlan()
        vision_outputs = plan_exec.run_modality_encoder(
            module=modality_encoder, pixel_values=ExecutionFuture("pixel_values")
        )
        merged = plan_exec.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={"vision": IMAGE_TOKEN_ID},
            encoder_outputs={"vision": vision_outputs},
        )
        return plan_exec, plan_exec.run_language_model(module=language_model, inputs=merged)

    exec_plan, output_future = build_plan()
    schedule = ctx.create_schedule(
        exec_plan,
        output_future,
        num_microbatches=num_microbatches if llm_pp > 1 else 1,
        microbatch_size=batch_size // num_microbatches if llm_pp > 1 else 1,
    )

    params = list(language_model.parameters()) + list(modality_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    train_loop(ctx, schedule, loader, optimizer, causal_lm_criterion, steps)


if __name__ == "__main__":
    tyro.cli(main)
