"""Cross-mesh (disjoint-rank) multimodal training through the Option C surface.

These tests cover the regression the language-only integration suite cannot: a
VLM whose vision encoder and language model are parallelized onto **disjoint**
rank ranges.  ``NonPipelineParallelSchedule`` runs the whole DAG on every rank,
which fails here because the vision encoder is materialized only on the vision
ranks (and stays ``meta`` on the language-model ranks).  ``CompiledSchedule``
runs each node only on its owning mesh and compiles the cross-mesh transfer that
moves the projected vision features to the ranks that consume them.

Everything runs on CPU under gloo (``world_size <= 4``) so the suite stays cheap
and offline.
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


def _make_batch(vision_config, vocab_size: int, batch_size: int = 2):
    num_image_tokens = _num_vision_tokens(vision_config)
    seq_len = num_image_tokens + 3
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), generator=g)
    input_ids[:, :num_image_tokens] = IMAGE_TOKEN_ID
    image_size = vision_config.image_size
    pixel_values = torch.randn(
        batch_size, 3, image_size, image_size, generator=g
    )
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "pixel_values": pixel_values,
    }


def _build_vlm_plan(modality_encoder, language_model, batch):
    """Build the per-step plan exactly like ``examples/pretrain_vlm.py``."""
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=modality_encoder,
        pixel_values=batch["pixel_values"],
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        modality_token_ids={"vision": IMAGE_TOKEN_ID},
        encoder_outputs={"vision": vision_outputs},
    )
    language_outputs = plan.run_language_model(module=language_model, inputs=merged)
    return plan, language_outputs


def _build_text_only_plan(language_model, batch):
    """A plan with no vision node — the per-batch flexibility case."""
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        modality_token_ids={},
        encoder_outputs={},
    )
    language_outputs = plan.run_language_model(module=language_model, inputs=merged)
    return plan, language_outputs


def _criterion(output, batch):
    if isinstance(output, torch.Tensor):
        return output
    return output.loss if hasattr(output, "loss") else output["loss"]


def _has_grad(module) -> bool:
    return any(p.grad is not None for p in module.parameters())


class TestCrossMeshVLMStep(GlooDistributedTestBase):
    """A single forward+backward of a disjoint-rank VLM yields finite loss/grads."""

    @property
    def world_size(self) -> int:
        return 2

    def test_step(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()

        plan = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        plan.parallelize(
            modality_encoder,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        )
        plan.parallelize(
            language_model,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        )
        ctx = plan.materialize("cpu", dtype=torch.float32)
        language_model.train()
        modality_encoder.train()

        batch = _make_batch(vision_config, language_model.hf_config.vocab_size)

        exec_plan, output_future = _build_vlm_plan(
            modality_encoder, language_model, batch
        )
        schedule = ctx.create_schedule(exec_plan, output_future)

        optimizer = torch.optim.SGD(
            list(language_model.parameters()) + list(modality_encoder.parameters()),
            lr=0.01,
        )
        result = schedule.step(batch, _criterion, optimizer, return_loss=True)
        ctx.sync_gradients()

        rank = dist.get_rank()
        if rank == 0:
            # Vision mesh: produces features, no loss, but receives grad back.
            self.assertIsNone(result["loss"])
            self.assertTrue(_has_grad(modality_encoder), "vision got no grad")
        else:
            # Language-model mesh: owns the output, computes a finite loss.
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model), "language model got no grad")


class TestCrossMeshVLMTensorParallel(GlooDistributedTestBase):
    """Cross-mesh transfer broadcasts across the consumer's TP group and back.

    The vision encoder stays single-rank (CLIP has no registered TP plan) while
    the language model is tensor-parallel over two ranks, so the forward transfer
    must fan the features out to *both* LM ranks and the backward must collapse
    back to one producer.  Vision is rank 0; the LM is ranks 1 and 2.
    """

    @property
    def world_size(self) -> int:
        return 3

    def test_step(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()

        plan = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        plan.parallelize(
            modality_encoder,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        )
        plan.parallelize(
            language_model,
            ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=1, data_parallel_size=1),
        )
        ctx = plan.materialize("cpu", dtype=torch.float32)
        language_model.train()
        modality_encoder.train()

        batch = _make_batch(vision_config, language_model.hf_config.vocab_size)
        exec_plan, output_future = _build_vlm_plan(
            modality_encoder, language_model, batch
        )
        schedule = ctx.create_schedule(exec_plan, output_future)

        result = schedule.step(batch, _criterion, return_loss=True)

        rank = dist.get_rank()
        if rank == 0:
            self.assertIsNone(result["loss"])
            self.assertTrue(_has_grad(modality_encoder), "vision got no grad")
        else:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model), "language model got no grad")


class TestCrossMeshFlexibility(GlooDistributedTestBase):
    """Alternating image / text-only batches both train; vision idles on text."""

    @property
    def world_size(self) -> int:
        return 2

    def test_alternating_modalities(self) -> None:
        language_model, modality_encoder, vision_config = _build_models()

        plan = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        plan.parallelize(
            modality_encoder,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        )
        plan.parallelize(
            language_model,
            ParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        )
        ctx = plan.materialize("cpu", dtype=torch.float32)
        language_model.train()
        modality_encoder.train()

        optimizer = torch.optim.SGD(
            list(language_model.parameters()) + list(modality_encoder.parameters()),
            lr=0.01,
        )
        rank = dist.get_rank()
        vocab = language_model.hf_config.vocab_size

        # --- Step 1: a batch WITH the vision modality. ---
        optimizer.zero_grad()
        batch = _make_batch(vision_config, vocab)
        exec_plan, output_future = _build_vlm_plan(
            modality_encoder, language_model, batch
        )
        schedule = ctx.create_schedule(exec_plan, output_future)
        schedule.step(batch, _criterion, optimizer)
        if rank == 0:
            self.assertTrue(_has_grad(modality_encoder), "vision should train on image step")

        # --- Step 2: a batch WITHOUT the vision modality. ---
        optimizer.zero_grad()
        text_batch = {
            k: v for k, v in batch.items() if k != "pixel_values"
        }
        exec_plan, output_future = _build_text_only_plan(language_model, text_batch)
        schedule = ctx.create_schedule(exec_plan, output_future)
        result = schedule.step(text_batch, _criterion, optimizer)

        if rank == 0:
            # The vision mesh has no node this batch: it idles, no grad, no loss.
            self.assertIsNone(result["loss"])
            self.assertFalse(
                _has_grad(modality_encoder), "vision should idle on a text-only step"
            )
        else:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
            self.assertTrue(_has_grad(language_model))


if __name__ == "__main__":
    unittest.main()
