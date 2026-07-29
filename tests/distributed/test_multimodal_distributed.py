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
from unittest.mock import patch
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import clip_vision_config, llama_config

from cornstarch.distributed import (
    ParallelConfig,
    ParallelizationPlan,
    PipelinePartitionSpec,
)
from cornstarch.distributed.context_parallel.splitters import (
    HeadTailContextParallelSplitter,
    UniformContextParallelSplitter,
)
from cornstarch.distributed.cross_mesh_routing import (
    CP_MODALITY_MASKS_KEY,
    CP_ROUTING_OFFSETS_KEY,
    CrossMeshGroup,
    build_route_plan,
    route_autograd,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    build_fused_modality_encoder,
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


class TestCrossMeshCPRouting(GlooDistributedTestBase):
    """The CP seam routes only owned rows and applies the transpose in backward."""

    @property
    def world_size(self) -> int:
        return 4

    def test_vision_cp2_to_llm_cp4_forward_backward(self) -> None:
        rank = dist.get_rank()
        token = 99
        input_ids = torch.arange(16).unsqueeze(0)
        input_ids[:, 4:10] = token
        modality_mask = torch.ones(1, 6, dtype=torch.bool)
        source_offsets = UniformContextParallelSplitter().offsets_for_size(
            modality_mask, 2
        )
        destination_offsets = UniformContextParallelSplitter().offsets_for_size(
            input_ids, 4
        )
        group = CrossMeshGroup(
            ranks=(0, 1, 2, 3),
            producer_ranks=(0, 1),
            consumer_ranks=(0, 1, 2, 3),
            process_group=dist.group.WORLD,
            dp_rank=0,
            producer_tp_rank=0,
            producer_ep_rank=0,
            consumer_tp_rank=0,
            consumer_ep_rank=0,
        )
        plan = build_route_plan(
            global_input_ids=input_ids,
            token_id=token,
            source_attention_mask=modality_mask,
            source_offsets=source_offsets,
            destination_offsets=destination_offsets,
            seam_group=group,
            global_rank=rank,
        )
        if rank == 0:
            features = torch.tensor([[4.0], [5.0], [6.0]], requires_grad=True)
        elif rank == 1:
            features = torch.tensor([[7.0], [8.0], [9.0]], requires_grad=True)
        else:
            features = torch.empty((0, 1), requires_grad=True)

        # The routing path is only a variable all-to-all. A full-feature
        # gather/broadcast on either forward or backward is a test failure.
        with (
            patch.object(dist, "all_gather", side_effect=AssertionError("full gather")),
            patch.object(dist, "broadcast", side_effect=AssertionError("broadcast")),
        ):
            received = route_autograd(features, plan, dist.group.WORLD)
            (received * float(rank + 1)).sum().backward()

        expected = {
            0: torch.empty((0, 1)),
            1: torch.tensor([[4.0], [5.0], [6.0], [7.0]]),
            2: torch.tensor([[8.0], [9.0]]),
            3: torch.empty((0, 1)),
        }[rank]
        torch.testing.assert_close(received, expected)
        if rank == 0:
            torch.testing.assert_close(features.grad, torch.full_like(features, 2.0))
        elif rank == 1:
            torch.testing.assert_close(
                features.grad, torch.tensor([[2.0], [3.0], [3.0]])
            )

    def test_one_to_many_many_to_one_and_headtail(self) -> None:
        rank = dist.get_rank()
        token = 99
        input_ids = torch.full((1, 8), token, dtype=torch.long)
        modality_mask = torch.ones(1, 8, dtype=torch.bool)
        cases = (
            (1, 4, UniformContextParallelSplitter(), UniformContextParallelSplitter()),
            (4, 1, UniformContextParallelSplitter(), UniformContextParallelSplitter()),
            (2, 2, UniformContextParallelSplitter(), HeadTailContextParallelSplitter()),
        )
        for source_size, destination_size, source_splitter, destination_splitter in cases:
            source_offsets = source_splitter.offsets_for_size(
                modality_mask, source_size
            )
            destination_offsets = destination_splitter.offsets_for_size(
                input_ids, destination_size
            )
            group = CrossMeshGroup(
                ranks=(0, 1, 2, 3),
                producer_ranks=tuple(range(source_size)),
                consumer_ranks=tuple(range(destination_size)),
                process_group=dist.group.WORLD,
                dp_rank=0,
                producer_tp_rank=0,
                producer_ep_rank=0,
                consumer_tp_rank=0,
                consumer_ep_rank=0,
            )
            plan = build_route_plan(
                global_input_ids=input_ids,
                token_id=token,
                source_attention_mask=modality_mask,
                source_offsets=source_offsets,
                destination_offsets=destination_offsets,
                seam_group=group,
                global_rank=rank,
            )
            if rank < source_size:
                features = source_offsets[rank].float().unsqueeze(1).requires_grad_(True)
            else:
                features = torch.empty((0, 1), requires_grad=True)
            received = route_autograd(features, plan, dist.group.WORLD)
            (received * float(rank + 1)).sum().backward()

            expected_received = (
                destination_offsets[rank].float().unsqueeze(1)
                if rank < destination_size
                else torch.empty((0, 1))
            )
            torch.testing.assert_close(received, expected_received)
            if rank < source_size:
                destination_owner = {
                    int(position): cp_rank
                    for cp_rank, offsets in enumerate(destination_offsets)
                    for position in offsets.tolist()
                }
                expected_grad = torch.tensor(
                    [
                        [float(destination_owner[int(position)] + 1)]
                        for position in source_offsets[rank].tolist()
                    ]
                )
                torch.testing.assert_close(features.grad, expected_grad)


class TestCrossMeshCPCompiledSchedule(GlooDistributedTestBase):
    """An actual ParallelContext routes CP2 projected rows into an LLM CP4."""

    @property
    def world_size(self) -> int:
        return 6

    def test_variable_count_loss_and_gradient_parity(self) -> None:
        language_model, modality_encoder, _ = _build_models()
        source_splitter = UniformContextParallelSplitter()
        destination_splitter = UniformContextParallelSplitter()
        parallel = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        parallel.parallelize(
            modality_encoder,
            ParallelConfig(
                pipeline_parallel_size=1,
                context_parallel_size=2,
                context_parallel_splitter=source_splitter,
            ),
        )
        parallel.parallelize(
            language_model,
            ParallelConfig(
                pipeline_parallel_size=1,
                context_parallel_size=4,
                context_parallel_splitter=destination_splitter,
            ),
        )
        ctx = parallel.materialize("cpu", dtype=torch.float32)

        token = IMAGE_TOKEN_ID
        input_ids = torch.tensor(
            [
                [token, token, token, 1, 2, 3, token, token, token, 4, 5, 6],
                [7, token, token, 8, 9, token, 10, 11, token, 12, 13, 14],
            ]
        )
        labels = input_ids.clone()
        modality_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        source_offsets = source_splitter.offsets_for_size(modality_mask, 2)
        destination_offsets = destination_splitter.offsets_for_size(input_ids, 4)
        vision_hidden = modality_encoder.projector.config.in_features
        language_hidden = modality_encoder.projector.config.out_features
        raw_full = (
            torch.arange(2 * 6 * vision_hidden, dtype=torch.float32)
            .reshape(2, 6, vision_hidden)
            .div(100.0)
        )

        rank = dist.get_rank()
        if rank < 2:
            local_raw = raw_full[:, source_offsets[rank]].clone().requires_grad_(True)

            def projected_forward(self, modality_features):
                return self.projector(modality_features)

            modality_encoder.forward = MethodType(projected_forward, modality_encoder)
            proj = modality_encoder.projector.projection
            with torch.no_grad():
                proj.weight.copy_(
                    torch.arange(proj.weight.numel()).reshape_as(proj.weight).div(10000)
                )
                proj.bias.copy_(torch.arange(proj.bias.numel()).div(10000))
        else:
            local_raw = torch.empty(0)
            cp_rank = rank - 2

            def local_language_forward(
                self, input_ids=None, inputs_embeds=None, labels=None, **kwargs
            ):
                logits = self.post_decoder["lm_head"](inputs_embeds)
                self._test_local_logits = logits.detach()
                return SimpleNamespace(loss=logits.square().sum(), logits=logits)

            language_model.forward = MethodType(
                local_language_forward, language_model
            )
            embed = language_model.pre_decoder["embed_tokens"].weight
            head = language_model.post_decoder["lm_head"].weight
            with torch.no_grad():
                embed.copy_(
                    torch.arange(embed.numel()).reshape_as(embed).div(100000)
                )
                head.copy_(torch.arange(head.numel()).reshape_as(head).div(100000))

        # An analytical non-CP reference uses the same deterministic seam
        # parameters, full text sequence, and padded modality batch.
        proj_weight = (
            torch.arange(language_hidden * vision_hidden, dtype=torch.float32)
            .reshape(language_hidden, vision_hidden)
            .div(10000)
            .requires_grad_(True)
        )
        proj_bias = (
            torch.arange(language_hidden, dtype=torch.float32)
            .div(10000)
            .requires_grad_(True)
        )
        vocab_size = language_model.hf_config.vocab_size
        embed_weight = (
            torch.arange(vocab_size * language_hidden, dtype=torch.float32)
            .reshape(vocab_size, language_hidden)
            .div(100000)
            .requires_grad_(True)
        )
        head_weight = embed_weight.detach().clone().requires_grad_(True)
        reference_raw = raw_full.clone().requires_grad_(True)
        projected = F.linear(reference_raw, proj_weight, proj_bias)
        safe_ids = input_ids.masked_fill(input_ids == token, 0)
        reference_embeds = F.embedding(safe_ids, embed_weight)
        for batch_index in range(input_ids.shape[0]):
            reference_embeds[batch_index, input_ids[batch_index] == token] = projected[
                batch_index, modality_mask[batch_index]
            ]
        reference_logits = F.linear(reference_embeds, head_weight)
        reference_loss = reference_logits.square().sum()
        reference_loss.backward()

        if rank >= 2:
            offsets = destination_offsets[cp_rank]
            local_ids = input_ids[:, offsets].contiguous()
            local_labels = labels[:, offsets].contiguous()
            local_positions = offsets.unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            local_ids = input_ids
            local_labels = labels
            local_positions = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(
                input_ids.shape[0], -1
            )
        microbatch = {
            "input_ids": local_ids,
            "labels": local_labels,
            "position_ids": local_positions,
            "cp_global_input_ids": input_ids,
            CP_MODALITY_MASKS_KEY: {"vision": modality_mask},
            CP_ROUTING_OFFSETS_KEY: {
                id(modality_encoder): tuple(source_offsets),
                id(language_model): tuple(destination_offsets),
            },
            "modality_features": local_raw,
        }
        exec_plan = CornstarchExecutionPlan()
        vision_outputs = exec_plan.run_modality_encoder(
            module=modality_encoder,
            modality_features=ExecutionFuture("modality_features"),
        )
        merged = exec_plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={"vision": token},
            encoder_outputs={"vision": vision_outputs},
            language_model_inputs={"position_ids": ExecutionFuture("position_ids")},
        )
        output = exec_plan.run_language_model(language_model, merged)
        with (
            patch.object(dist, "all_gather", side_effect=AssertionError("full gather")),
            patch.object(dist, "broadcast", side_effect=AssertionError("broadcast")),
        ):
            result = ctx.create_schedule(exec_plan, output).step(
                [microbatch], _criterion, return_loss=True
            )
            ctx.sync_gradients()

        distributed_loss = torch.zeros(1)
        if result["loss"] is not None:
            distributed_loss.copy_(result["loss"])
        dist.all_reduce(distributed_loss)
        torch.testing.assert_close(distributed_loss.squeeze(), reference_loss.detach())
        if rank < 2:
            proj = modality_encoder.projector.projection
            torch.testing.assert_close(proj.weight.grad, proj_weight.grad)
            torch.testing.assert_close(proj.bias.grad, proj_bias.grad)
            torch.testing.assert_close(
                local_raw.grad, reference_raw.grad[:, source_offsets[rank]]
            )
        else:
            torch.testing.assert_close(
                language_model._test_local_logits,
                reference_logits.detach()[:, destination_offsets[cp_rank]],
            )
            torch.testing.assert_close(
                language_model.pre_decoder["embed_tokens"].weight.grad,
                embed_weight.grad,
            )
            torch.testing.assert_close(
                language_model.post_decoder["lm_head"].weight.grad,
                head_weight.grad,
            )


if __name__ == "__main__":
    unittest.main()


class TestFusedEncoderAndLanguagePipeline(GlooDistributedTestBase):
    """Two fused encoders span PP=2 before an independently PP=2 LLM."""

    @property
    def world_size(self) -> int:
        return 4

    def test_fused_pp2_to_llm_pp2_forward_backward(self) -> None:
        vision_config = _tiny_vision_config()
        language_model = from_hf_config(
            llama_config(), model_kind="language", attn_implementation="eager"
        )
        children = {}
        for modality in ("vision", "aux"):
            encoder = from_hf_config(
                vision_config, model_kind="vision", attn_implementation="eager"
            )
            children[modality] = build_modality_encoder(
                encoder, language_model, modality=modality
            )
        fused = build_fused_modality_encoder(children)
        language_model.set_random_init()
        fused.set_random_init()

        parallel = ParallelizationPlan(global_ranks=list(range(self.world_size)))
        encoder_partition = PipelinePartitionSpec((1, 2))
        parallel.parallelize(
            fused,
            ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                data_parallel_size=1,
            ),
            pipeline_partitions={
                "vision": encoder_partition,
                "aux": encoder_partition,
            },
        )
        llm_layers = language_model.hf_config.num_hidden_layers
        parallel.parallelize(
            language_model,
            ParallelConfig(
                tensor_parallel_size=1,
                pipeline_parallel_size=2,
                data_parallel_size=1,
            ),
            pipeline_partitions=PipelinePartitionSpec((1, llm_layers)),
        )
        context = parallel.materialize("cpu", dtype=torch.float32)

        tokens_per_encoder = _num_vision_tokens(vision_config)
        seq_len = tokens_per_encoder * 2 + 3
        generator = torch.Generator().manual_seed(7)
        input_ids = torch.randint(
            2,
            language_model.hf_config.vocab_size,
            (4, seq_len),
            generator=generator,
        )
        input_ids[:, :tokens_per_encoder] = 0
        input_ids[:, tokens_per_encoder : 2 * tokens_per_encoder] = 1
        image_size = vision_config.image_size
        batch = {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "vision_pixels": torch.randn(
                4, 3, image_size, image_size, generator=generator
            ),
            "aux_pixels": torch.randn(
                4, 3, image_size, image_size, generator=generator
            ),
        }
        microbatches = [
            {key: value.chunk(4, dim=0)[index] for key, value in batch.items()}
            for index in range(4)
        ]
        microbatches[1].pop("aux_pixels")
        microbatches[1]["input_ids"].masked_fill_(
            microbatches[1]["input_ids"] == 1, 2
        )
        microbatches[1]["labels"] = microbatches[1]["input_ids"].clone()
        microbatches[2].pop("vision_pixels")
        microbatches[2]["input_ids"].masked_fill_(
            microbatches[2]["input_ids"] == 0, 2
        )
        microbatches[2]["labels"] = microbatches[2]["input_ids"].clone()
        microbatches[3].pop("vision_pixels")
        microbatches[3].pop("aux_pixels")
        microbatches[3]["input_ids"].masked_fill_(
            (microbatches[3]["input_ids"] == 0)
            | (microbatches[3]["input_ids"] == 1),
            2,
        )
        microbatches[3]["labels"] = microbatches[3]["input_ids"].clone()


        execution = CornstarchExecutionPlan()
        fused_outputs = execution.run_fused_modality_encoder(
            fused,
            inputs={
                "vision": {
                    "pixel_values": ExecutionFuture("vision_pixels", optional=True)
                },
                "aux": {
                    "pixel_values": ExecutionFuture("aux_pixels", optional=True)
                },
            },
        )
        merged = execution.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={"vision": 0, "aux": 1},
            encoder_outputs=fused_outputs,
        )
        output = execution.run_language_model(language_model, inputs=merged)
        schedule = context.create_schedule(execution, output)
        self.assertEqual(schedule.num_stages, 4)
        result = schedule.step(microbatches, _criterion, return_loss=True)

        rank = dist.get_rank()
        if rank == 3:
            self.assertIsNotNone(result["loss"])
            self.assertTrue(result["loss"].isfinite().all())
        else:
            self.assertIsNone(result["loss"])
        if rank in (0, 1):
            self.assertTrue(_has_grad(fused), "fused encoder stage got no grad")
        else:
            self.assertTrue(_has_grad(language_model), "language stage got no grad")
