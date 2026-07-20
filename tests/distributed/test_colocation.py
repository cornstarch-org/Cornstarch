"""Co-located (no-pipeline-parallel) materialization through the Option C surface.

When every ``ParallelConfig.pipeline_parallel_size`` is ``None`` the modules are
**co-located**: every rank materializes every modality and runs the whole
``encoder -> merge -> language model`` plan locally (no cross-rank transfer, no
schedule).  Data parallelism replicates that across ranks.  These tests pin that
behavior; the disaggregated (pipeline-parallel) path is covered by
``test_parallelism_integration`` / ``test_multimodal_distributed``.
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


def _is_materialized(module) -> bool:
    return not any(p.is_meta for p in module.parameters())


def _has_grad(module) -> bool:
    return any(p.grad is not None for p in module.parameters())


class TestColocatedLanguageModel(GlooDistributedTestBase):
    """A no-PP LM is co-located + data-parallel: every rank holds the whole model."""

    @property
    def world_size(self) -> int:
        return 2

    def test_step(self) -> None:
        model = from_hf_config(
            llama_config(), model_kind="language", attn_implementation="eager"
        )
        model.set_random_init()

        plan = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        plan.parallelize(model, ParallelConfig(data_parallel_size=self.world_size))
        ctx = plan.materialize("cpu", dtype=torch.float32)

        self.assertFalse(ctx.uses_pipeline_parallel)
        self.assertTrue(_is_materialized(model), "model should be real on every rank")
        model.train()

        torch.manual_seed(dist.get_rank())
        vocab = model.hf_config.vocab_size
        batch = {
            "input_ids": torch.randint(0, vocab, (2, 16)),
            "labels": torch.randint(0, vocab, (2, 16)),
        }

        exec_plan = CornstarchExecutionPlan()
        merged = exec_plan.merge_modality_encoder_outputs(
            language_model=model,
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            modality_token_ids={},
            encoder_outputs={},
        )
        output_future = exec_plan.run_language_model(module=model, inputs=merged)

        # No schedule: run the plan directly on this rank, like the
        # non-distributed example.
        output = output_future.execute()
        loss = output.loss
        loss.backward()
        ctx.sync_gradients()

        self.assertTrue(loss.isfinite().all())
        self.assertTrue(_has_grad(model))


class TestColocatedVLM(GlooDistributedTestBase):
    """A no-PP VLM co-locates encoder + LM on every rank and runs locally."""

    @property
    def world_size(self) -> int:
        return 2

    def test_step(self) -> None:
        vision_config = clip_vision_config()
        vision_config.image_size = 8
        vision_config.patch_size = 2
        vision_config.num_hidden_layers = 2

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

        plan = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        # All pipeline_parallel_size=None -> co-located on the same ranks.
        plan.parallelize(
            modality_encoder, ParallelConfig(data_parallel_size=self.world_size)
        )
        plan.parallelize(
            language_model, ParallelConfig(data_parallel_size=self.world_size)
        )
        ctx = plan.materialize("cpu", dtype=torch.float32)

        self.assertFalse(ctx.uses_pipeline_parallel)
        # Both modalities live on every rank.
        self.assertTrue(_is_materialized(language_model))
        self.assertTrue(_is_materialized(modality_encoder))
        language_model.train()
        modality_encoder.train()

        torch.manual_seed(dist.get_rank())
        vocab = language_model.hf_config.vocab_size
        num_image_tokens = (vision_config.image_size // vision_config.patch_size) ** 2 + 1
        input_ids = torch.randint(1, vocab, (2, num_image_tokens + 3))
        input_ids[:, :num_image_tokens] = IMAGE_TOKEN_ID
        batch = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "pixel_values": torch.randn(
                2, 3, vision_config.image_size, vision_config.image_size
            ),
        }

        exec_plan = CornstarchExecutionPlan()
        vision_outputs = exec_plan.run_modality_encoder(
            module=modality_encoder, pixel_values=batch["pixel_values"]
        )
        merged = exec_plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            modality_token_ids={"vision": IMAGE_TOKEN_ID},
            encoder_outputs={"vision": vision_outputs},
        )
        output_future = exec_plan.run_language_model(
            module=language_model, inputs=merged
        )

        # Co-located: the whole encoder -> merge -> LM plan runs locally; no
        # cross-rank transfer, no schedule.
        output = output_future.execute()
        loss = output.loss
        loss.backward()
        ctx.sync_gradients()

        self.assertTrue(loss.isfinite().all())
        self.assertTrue(_has_grad(language_model))
        self.assertTrue(_has_grad(modality_encoder))


if __name__ == "__main__":
    unittest.main()
