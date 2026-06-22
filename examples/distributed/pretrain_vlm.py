"""Distributed VLM pretraining via the Option C per-modality surface.

Run with torchrun, e.g. 4 GPUs with the vision encoder tensor-parallel and the
language model tensor + pipeline parallel, all replicated once (dp=1)::

    torchrun --nproc_per_node=4 examples/distributed/pretrain_vlm.py \
        --vision-tp 2 --llm-tp 2 --llm-pp 1

This reads top-to-bottom like the non-distributed ``examples/pretrain_vlm.py``:
build the modality encoder with ``build_modality_encoder``, set its init plan,
build the execution plan, and run an explicit training loop.  The only
distributed-specific lines are ``init_distributed()``, the per-modality
``plan.parallelize`` + ``plan.materialize``, ``ctx.prepare_dataloader``,
``ctx.create_schedule`` + ``schedule.step``, and ``ctx.sync_gradients``.

Each modality is described independently: ``plan.parallelize`` is called once
per modality with its own ``ParallelConfig`` (vision can be TP-only while the
LLM is TP+PP), and ``plan.materialize`` resolves the per-modality + DP-offset
rank math and materializes both models.  The projector follows its encoder
automatically — there is no standalone projector setup in this script.
"""
from __future__ import annotations

import tyro

import torch
from torch.utils.data import Dataset
from transformers import AutoConfig

from common import DTYPE, causal_lm_criterion, init_distributed

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    build_modality_encoder,
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
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )

    language_model.set_random_init()
    modality_encoder.set_random_init()

    # Per-modality declarative configs — vision and the LLM are parallelized
    # independently; materialize() handles the cross-modality rank assignment and
    # materializes each modality encoder's projector alongside its encoder.
    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        modality_encoder,
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
    ctx = plan.materialize(device, dtype=DTYPE)
    language_model.train()
    modality_encoder.train()

    patches_per_side = int(vision_config.image_size) // int(vision_config.patch_size)
    num_image_tokens = patches_per_side * patches_per_side + 1  # + CLS token
    image_size = (int(vision_config.image_size), int(vision_config.image_size))
    dataset = FakeVLMDataset(
        language_model.hf_config.vocab_size, seq_len, num_image_tokens, image_size
    )
    loader = ctx.prepare_dataloader(dataset, batch_size=batch_size, shuffle=True)

    def build_plan():
        exec_plan = CornstarchExecutionPlan()
        vision_outputs = exec_plan.run_modality_encoder(
            module=modality_encoder, pixel_values=ExecutionFuture("pixel_values")
        )
        merged = exec_plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={"vision": IMAGE_TOKEN_ID},
            encoder_outputs={"vision": vision_outputs},
        )
        output_future = exec_plan.run_language_model(
            module=language_model, inputs=merged
        )
        return exec_plan, output_future

    exec_plan, output_future = build_plan()
    schedule = ctx.create_schedule(
        exec_plan,
        output_future,
        num_microbatches=num_microbatches if llm_pp > 1 else 1,
        microbatch_size=batch_size // num_microbatches if llm_pp > 1 else 1,
    )

    params = list(language_model.parameters()) + list(modality_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
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
