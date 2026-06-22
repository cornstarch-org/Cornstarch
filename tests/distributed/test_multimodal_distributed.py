"""Disaggregated (pipeline-parallel) multimodal training: encoder as a leading stage.

When pipeline parallelism is used, a VLM's vision encoder and language model are
disaggregated onto disjoint ranks and form one pipeline: the encoder is the
**leading stage**, feeding the language-model stages across the cross-mesh seam.
``OneForwardOneBackwardSchedule`` drives it, microbatched from the user's list.

These run on CPU under gloo (``world_size <= 4``). The co-located (no-PP) VLM
path is covered by ``test_colocation``; the LM-only pipeline by
``test_parallelism_integration``.
"""
from __future__ import annotations

import unittest

import torch
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import clip_vision_config, llama_config

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    build_modality_encoder,
    from_hf_config,
)

IMAGE_TOKEN_ID = 0


def _tiny_vision_config():
    config = clip_vision_config()
    config.image_size = 8
    config.patch_size = 2
    config.num_hidden_layers = 2
    return config


def _build_models():
    vision_config = _tiny_vision_config()
    language_model = from_hf_config(
        llama_config(), model_kind="language", attn_implementation="eager"
    )
    vision_encoder = from_hf_config(
        vision_config, model_kind="vision", attn_implementation="eager"
    )
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )
    language_model.set_random_init()
    modality_encoder.set_random_init()
    return language_model, modality_encoder, vision_config


def _num_vision_tokens(vision_config) -> int:
    return (vision_config.image_size // vision_config.patch_size) ** 2 + 1


def _make_microbatches(vision_config, vocab_size, batch_size, num_microbatches):
    """Build a batch and split it into ``num_microbatches`` (one image per sample)."""
    num_image_tokens = _num_vision_tokens(vision_config)
    seq_len = num_image_tokens + 3
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), generator=g)
    input_ids[:, :num_image_tokens] = IMAGE_TOKEN_ID
    image = vision_config.image_size
    pixel_values = torch.randn(batch_size, 3, image, image, generator=g)
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "pixel_values": pixel_values,
    }
    return [
        {k: v.chunk(num_microbatches, dim=0)[i] for k, v in batch.items()}
        for i in range(num_microbatches)
    ]


def _vlm_plan(language_model, modality_encoder):
    """A futures-based plan so the schedule can feed each microbatch in turn."""
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=modality_encoder, pixel_values=ExecutionFuture("pixel_values")
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={"vision": IMAGE_TOKEN_ID},
        encoder_outputs={"vision": vision_outputs},
    )
    output_future = plan.run_language_model(module=language_model, inputs=merged)
    return plan, output_future


def _criterion(output, batch):
    if isinstance(output, torch.Tensor):
        return output
    return output.loss if hasattr(output, "loss") else output["loss"]


def _has_grad(module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def _parallelize(modality_encoder, language_model, world_size, *, vision_tp, llm_tp, llm_pp):
    plan = ParallelizationPlan(global_ranks=list(range(world_size)))
    plan.parallelize(
        modality_encoder,
        ParallelConfig(
            tensor_parallel_size=vision_tp,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
    )
    plan.parallelize(
        language_model,
        ParallelConfig(
            tensor_parallel_size=llm_tp,
            pipeline_parallel_size=llm_pp,
            data_parallel_size=1,
        ),
    )
    return plan.materialize("cpu", dtype=torch.float32)


class TestCrossMeshVLMStep(GlooDistributedTestBase):
    """Encoder (rank 0) feeds a single-stage language model (rank 1)."""

    @property
    def world_size(self) -> int:
        return 2

    def test_step(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()
        ctx = _parallelize(
            modality_encoder, language_model, self.world_size,
            vision_tp=1, llm_tp=1, llm_pp=1,
        )
        self.assertTrue(ctx.uses_pipeline_parallel)
        language_model.train()
        modality_encoder.train()

        microbatches = _make_microbatches(
            vision_config, language_model.hf_config.vocab_size,
            batch_size=2, num_microbatches=2,
        )
        plan, output_future = _vlm_plan(language_model, modality_encoder)
        schedule = ctx.create_schedule(plan, output_future)
        result = schedule.step(microbatches, _criterion, return_loss=True)
        ctx.sync_gradients()

        if dist.get_rank() == 0:
            self.assertIsNone(result["loss"])
            self.assertTrue(_has_grad(modality_encoder), "vision got no grad")
        else:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model), "language model got no grad")


class TestCrossMeshVLMPipeline(GlooDistributedTestBase):
    """Encoder (rank 0) + a 2-stage pipelined language model (ranks 1, 2)."""

    @property
    def world_size(self) -> int:
        return 3

    def test_step(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()
        ctx = _parallelize(
            modality_encoder, language_model, self.world_size,
            vision_tp=1, llm_tp=1, llm_pp=2,
        )
        language_model.train()
        modality_encoder.train()

        microbatches = _make_microbatches(
            vision_config, language_model.hf_config.vocab_size,
            batch_size=4, num_microbatches=2,
        )
        plan, output_future = _vlm_plan(language_model, modality_encoder)
        schedule = ctx.create_schedule(plan, output_future)
        result = schedule.step(microbatches, _criterion, return_loss=True)
        ctx.sync_gradients()

        rank = dist.get_rank()
        if rank == 2:  # last LM stage
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
        else:  # encoder (0) and first LM stage (1) hold no loss
            self.assertIsNone(result["loss"])

        if rank == 0:
            self.assertTrue(_has_grad(modality_encoder), "vision got no grad")
        else:
            self.assertTrue(_has_grad(language_model), "language model got no grad")


class TestCrossMeshVLMTensorParallel(GlooDistributedTestBase):
    """Seam broadcasts to the consumer's TP group: encoder (0) -> LM tp=2 (1, 2)."""

    @property
    def world_size(self) -> int:
        return 3

    def test_step(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()
        ctx = _parallelize(
            modality_encoder, language_model, self.world_size,
            vision_tp=1, llm_tp=2, llm_pp=1,
        )
        language_model.train()
        modality_encoder.train()

        microbatches = _make_microbatches(
            vision_config, language_model.hf_config.vocab_size,
            batch_size=2, num_microbatches=1,
        )
        plan, output_future = _vlm_plan(language_model, modality_encoder)
        schedule = ctx.create_schedule(plan, output_future)
        result = schedule.step(microbatches, _criterion, return_loss=True)

        if dist.get_rank() == 0:
            self.assertIsNone(result["loss"])
            self.assertTrue(_has_grad(modality_encoder), "vision got no grad")
        else:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model), "language model got no grad")


class TestCrossMeshFlexibility(GlooDistributedTestBase):
    """Image steps run encoder+LM; text-only steps run LM only with vision idle."""

    @property
    def world_size(self) -> int:
        return 2

    def test_alternating_modalities(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()
        ctx = _parallelize(
            modality_encoder, language_model, self.world_size,
            vision_tp=1, llm_tp=1, llm_pp=1,
        )
        language_model.train()
        modality_encoder.train()
        optimizer = torch.optim.SGD(
            list(language_model.parameters()) + list(modality_encoder.parameters()),
            lr=0.01,
        )
        rank = dist.get_rank()
        vocab = language_model.hf_config.vocab_size

        # Image step: encoder is the leading stage.
        optimizer.zero_grad()
        microbatches = _make_microbatches(vision_config, vocab, 2, 2)
        plan, output_future = _vlm_plan(language_model, modality_encoder)
        ctx.create_schedule(plan, output_future).step(microbatches, _criterion, optimizer)
        if rank == 0:
            self.assertTrue(_has_grad(modality_encoder), "vision should train on images")

        # Text-only step: the plan has no encoder node, so the vision ranks idle
        # and the language-model stage runs alone.
        optimizer.zero_grad()
        text_mbs = [{k: v for k, v in mb.items() if k != "pixel_values"} for mb in microbatches]
        plan = CornstarchExecutionPlan()
        merged = plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={},
            encoder_outputs={},
        )
        output_future = plan.run_language_model(module=language_model, inputs=merged)
        result = ctx.create_schedule(plan, output_future).step(text_mbs, _criterion, optimizer)

        if rank == 0:
            self.assertIsNone(result["loss"])
            self.assertFalse(_has_grad(modality_encoder), "vision should idle on text")
        else:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model))


if __name__ == "__main__":
    unittest.main()
