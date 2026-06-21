"""Single-GPU-vs-parallel numerical equivalence tests.

For each model-side parallelism (TP, PP, EP) and for data parallelism (DP) we
build a non-parallel reference, run a forward+backward, then build the parallel
model with *identical* weights and assert the loss (and a representative
gradient) matches the reference.  Parallelism is mathematically exact, but
bf16 accumulation differs slightly across the sharded collectives, so the
tolerance is ``atol=rtol=1e-3`` per the project's numerical-equivalence
decision.

Weights are made identical by copying the reference parameters into the
parallel model after materialization (DTensor params receive the correctly
sharded slice via ``distribute_tensor``).  This is test-only setup — it does
not rely on sharded checkpoint load, which is intentionally out of scope.

All tests use the gloo backend so they run on CPU.  Context parallelism's
all-gather flash-attention kernel is CUDA-only, so its numerical-equivalence
test is guarded to require >=2 visible CUDA devices and otherwise skips.
"""
from __future__ import annotations

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import llama_config, qwen3_5_moe_config

from cornstarch.distributed.data_parallel import allreduce_gradients
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import (
    PipelineParallelForwardSpec,
)
from cornstarch.distributed.pipeline_parallel.schedule import (
    OneForwardOneBackwardSchedule,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.tensor_parallel import apply_tensor_parallel
from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture, from_hf_config


DTYPE = torch.bfloat16
ATOL = 1e-3
RTOL = 1e-3
VOCAB = 128


def _build_llm(dtype: torch.dtype = DTYPE, moe: bool = False):
    """Build a tiny LM with deterministic weights (seed reset before build)."""
    torch.manual_seed(0)
    config = qwen3_5_moe_config() if moe else llama_config()
    config.vocab_size = VOCAB
    config.tie_word_embeddings = False
    model = from_hf_config(config, model_kind="language", attn_implementation="eager")
    model.set_random_init()
    model.materialize("cpu", dtype=dtype)
    model.train()
    return model


def _ref_params(model: nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot full (non-sharded) parameters keyed by name."""
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _copy_full_into(model: nn.Module, ref: dict[str, torch.Tensor]) -> None:
    """Copy reference params into ``model``, sharding DTensor params as needed."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            src = ref[name]
            if isinstance(p.data, DTensor):
                sharded = distribute_tensor(src, p.data.device_mesh, p.data.placements)
                p.data.copy_(sharded)
            else:
                p.data.copy_(src)


def _batch(batch_size: int = 4, seq_len: int = 16, seed: int = 1234):
    """A reproducible batch identical on every rank (same seed)."""
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (batch_size, seq_len), generator=g)
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _loss(model: nn.Module, batch: dict) -> torch.Tensor:
    return model(input_ids=batch["input_ids"], labels=batch["labels"]).loss


class TestTensorParallelEquivalence(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_tp_matches_single(self):
        mesh = ModalProcessGroupMesh(
            device_type="cpu", global_ranks=[0, 1],
            dp_size=1, cp_size=1, tp_size=2, num_pp_stages=1,
        )

        ref_model = _build_llm()
        batch = _batch()
        ref_loss = _loss(ref_model, batch)
        ref_loss.backward()
        ref_embed_grad = ref_model.pre_decoder["embed_tokens"].weight.grad.clone()
        ref = _ref_params(ref_model)

        model = _build_llm()
        apply_tensor_parallel(model, mesh.tp_mesh)
        # apply_tensor_parallel records DTensor specs on already-materialized
        # params here; re-materialize is not needed because parallelize_module
        # converts them in place. Copy identical weights into the shards.
        _copy_full_into(model, ref)

        tp_loss = _loss(model, batch)
        torch.testing.assert_close(tp_loss, ref_loss, atol=ATOL, rtol=RTOL)

        tp_loss.backward()
        embed_grad = model.pre_decoder["embed_tokens"].weight.grad
        torch.testing.assert_close(embed_grad, ref_embed_grad, atol=ATOL, rtol=RTOL)


class TestPipelineParallelEquivalence(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_pp_matches_single(self):
        mesh = ModalProcessGroupMesh(
            device_type="cpu", global_ranks=[0, 1],
            dp_size=1, cp_size=1, tp_size=1, num_pp_stages=2,
        )

        ref_model = _build_llm()
        batch = _batch(batch_size=4)
        ref_loss = _loss(ref_model, batch)
        ref = _ref_params(ref_model)

        # Build the stage-local model: slice layers, wrap the forward spec.
        model = _build_llm()
        total = len(model.decoder_layers)
        start, end = mesh.distribute_layers(total)
        model.decoder_layers = nn.ModuleList(list(model.decoder_layers)[start:end])
        model.forward_spec = PipelineParallelForwardSpec(model.forward_spec, mesh)

        # Copy identical weights (decoder layers are re-indexed per stage).
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.startswith("decoder_layers."):
                    _, idx, rest = name.split(".", 2)
                    src = ref[f"decoder_layers.{start + int(idx)}.{rest}"]
                else:
                    src = ref[name]
                p.data.copy_(src)

        plan = CornstarchExecutionPlan()
        merged = plan.merge_modality_encoder_outputs(
            language_model=model,
            input_ids=ExecutionFuture("input_ids"),
            labels=ExecutionFuture("labels"),
            modality_token_ids={},
            encoder_outputs={},
        )
        output_future = plan.run_language_model(module=model, inputs=merged)
        schedule = OneForwardOneBackwardSchedule(
            plan, output_future, mesh, num_microbatches=2, microbatch_size=2,
        )

        def criterion(output, micro_batch):
            if isinstance(output, torch.Tensor):
                return output
            return output.loss if hasattr(output, "loss") else output["loss"]

        result = schedule.step(batch, criterion, optimizer=None, return_loss=True)
        if mesh.is_last_stage():
            torch.testing.assert_close(
                result["loss"].reshape(()), ref_loss, atol=ATOL, rtol=RTOL
            )


class TestExpertParallelEquivalence(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_ep_matches_single(self):
        ep_group = dist.group.WORLD

        model = _build_llm(moe=True)
        batch = _batch(batch_size=2, seq_len=8)

        # Reference loss/grad on the replicated (pre-EP) model.
        model.eval()
        ref_loss = _loss(model, batch)
        model.train()
        ref_loss_train = _loss(model, batch)
        ref_loss_train.backward()
        router_grad_ref = model.decoder_layers[0].mlp.gate.weight.grad.clone()
        model.zero_grad()

        apply_expert_parallel(model, ep_group)

        model.eval()
        ep_loss = _loss(model, batch)
        torch.testing.assert_close(ep_loss, ref_loss, atol=ATOL, rtol=RTOL)

        model.train()
        ep_loss_train = _loss(model, batch)
        ep_loss_train.backward()
        # The router is replicated across EP ranks: its gradient must match.
        router_grad = model.decoder_layers[0].mlp.gate.weight.grad
        torch.testing.assert_close(
            router_grad, router_grad_ref, atol=ATOL, rtol=RTOL
        )


class TestDataParallelEquivalence(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_dp_grad_matches_full_batch(self):
        mesh = ModalProcessGroupMesh(
            device_type="cpu", global_ranks=[0, 1],
            dp_size=2, cp_size=1, tp_size=1, num_pp_stages=1,
        )
        rank = dist.get_rank()

        # Shared full batch; DP rank r processes its half.
        full = _batch(batch_size=4, seq_len=16)
        ref = _ref_params(_build_llm())

        # Reference: full-batch grads on a single model.
        ref_model = _build_llm()
        _loss(ref_model, full).backward()
        ref_grad = ref_model.pre_decoder["embed_tokens"].weight.grad.clone()

        # DP: each rank sees half, then gradients are all-reduced (averaged).
        model = _build_llm()
        _copy_full_into(model, ref)
        half = full["input_ids"].shape[0] // 2
        local = {
            "input_ids": full["input_ids"][rank * half : (rank + 1) * half],
            "labels": full["labels"][rank * half : (rank + 1) * half],
        }
        _loss(model, local).backward()
        allreduce_gradients(model, mesh.dp_group)

        dp_grad = model.pre_decoder["embed_tokens"].weight.grad
        torch.testing.assert_close(dp_grad, ref_grad, atol=ATOL, rtol=RTOL)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "context-parallel attention uses CUDA flash-attn and needs >=2 GPUs",
)
class TestContextParallelEquivalence(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def test_cp_matches_single(self):  # pragma: no cover - needs >=2 GPUs
        from cornstarch.distributed.context_parallel import apply_context_parallel
        from cornstarch.distributed.context_parallel.splitters import (
            UniformContextParallelSplitter,
        )

        mesh = ModalProcessGroupMesh(
            device_type="cuda", global_ranks=[0, 1],
            dp_size=1, cp_size=2, tp_size=1, num_pp_stages=1,
        )
        device = f"cuda:{torch.cuda.current_device()}"

        ref = _ref_params(_build_llm())
        ref_model = _build_llm().to(device)
        batch = _batch()
        batch = {k: v.to(device) for k, v in batch.items()}
        ref_loss = _loss(ref_model, batch)

        model = _build_llm()
        apply_context_parallel(model, mesh.cp_group)
        _copy_full_into(model, ref)
        model.to(device)

        splitter = UniformContextParallelSplitter()
        mask = torch.ones_like(batch["input_ids"], dtype=torch.float32)
        splitter.compute_offsets(mask, mesh.cp_group)
        local = {
            "input_ids": splitter.split(batch["input_ids"], mesh.cp_group),
            "labels": splitter.split(batch["labels"], mesh.cp_group),
        }
        cp_loss = _loss(model, local)
        # Per-rank CP loss is over the local shard; gather and average.
        losses = [torch.zeros_like(cp_loss) for _ in range(self.world_size)]
        dist.all_gather(losses, cp_loss.detach())
        torch.testing.assert_close(
            torch.stack(losses).mean(), ref_loss, atol=ATOL, rtol=RTOL
        )


if __name__ == "__main__":
    unittest.main()
