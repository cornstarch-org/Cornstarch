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
test runs the collectives on gloo (CPU, bridged via ``gloo_utils``) while the
flash-attention compute happens on the shared GPU.  Two gloo ranks share a
single ``cuda:0``, so the CP test runs on a one-GPU box (it only needs CUDA to
be available, not >=2 devices).  CP equivalence is asserted at the
attention-function level against a *non-causal* full-sequence SDPA reference,
because the CP kernel hardcodes ``causal=False`` (see ``tasks/backlog.md`` for
the deferred causal-LM CP equivalence).
"""
from __future__ import annotations

import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import llama_config, qwen3_5_moe_config

from cornstarch.distributed.data_parallel import allreduce_gradients
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import (
    PipelineParallelForwardSpec,
)
from cornstarch.distributed.pipeline_parallel.schedule import (
    MeshLayout,
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


class TestTensorParallelCheckpointInit(GlooDistributedTestBase):
    """Checkpoint init on a TP model shards weights correctly per rank.

    Drives the real materialize path: a meta model records DTensor specs via
    ``apply_tensor_parallel``, then ``set_checkpoint_init`` with a *full*
    (non-sharded) reference state dict materializes each rank's correctly-sharded
    slice — exactly ``distribute_tensor(full)`` — and reproduces the non-parallel
    loss.
    """

    @property
    def world_size(self) -> int:
        return 2

    def test_checkpoint_init_shards_weights(self):
        mesh = ModalProcessGroupMesh(
            device_type="cpu", global_ranks=[0, 1],
            dp_size=1, cp_size=1, tp_size=2, num_pp_stages=1,
        )

        # Non-parallel reference: full weights + loss/grad.
        ref_model = _build_llm()
        batch = _batch()
        ref_loss = _loss(ref_model, batch)
        ref_loss.backward()
        ref_embed_grad = ref_model.pre_decoder["embed_tokens"].weight.grad.clone()
        ref_params = _ref_params(ref_model)
        ref_hf_state_dict = ref_model.to_hf_state_dict()

        # Fresh meta model -> record DTensor specs -> checkpoint init -> materialize.
        torch.manual_seed(0)
        config = llama_config()
        config.vocab_size = VOCAB
        config.tie_word_embeddings = False
        model = from_hf_config(config, model_kind="language", attn_implementation="eager")
        apply_tensor_parallel(model, mesh.tp_mesh)
        model.set_checkpoint_init(state_dict=ref_hf_state_dict)
        model.materialize("cpu", dtype=DTYPE)
        model.train()

        # Every DTensor param holds this rank's distribute_tensor() slice; plain
        # params hold the full reference tensor.
        for name, p in model.named_parameters():
            full = ref_params[name]
            if isinstance(p.data, DTensor):
                expected = distribute_tensor(full, p.data.device_mesh, p.data.placements)
                torch.testing.assert_close(
                    p.data.to_local(), expected.to_local(), atol=ATOL, rtol=RTOL
                )
            else:
                torch.testing.assert_close(p.data, full, atol=ATOL, rtol=RTOL)

        # Loss/grad parity against the non-parallel reference.
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
        layout = MeshLayout(
            global_ranks=tuple(mesh._global_ranks),
            dp_size=mesh.dp_size,
            num_pp_stages=mesh.num_stages,
            cp_size=mesh.cp_size,
            tp_size=mesh.tp_size,
            ep_size=mesh.ep_size,
        )
        schedule = OneForwardOneBackwardSchedule(
            plan, output_future, {id(model): layout}, {id(model): mesh}, mesh.dp_size,
        )
        microbatches = [{k: v[i * 2 : i * 2 + 2] for k, v in batch.items()} for i in range(2)]

        def criterion(output, micro_batch):
            if isinstance(output, torch.Tensor):
                return output
            return output.loss if hasattr(output, "loss") else output["loss"]

        result = schedule.step(microbatches, criterion, optimizer=None, return_loss=True)
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


# CP flash-attn accumulates over an all-gathered K/V sequence in a different
# order than a single full-sequence SDPA matmul, so both forward and backward
# parity hold at 5e-3 — looser than the project-wide 1e-3 target but matching
# the legacy single-GPU CP flash-attn test (tests_old/.../test_context_parallel.py).
CP_ATOL = 5e-3
CP_RTOL = 5e-3


@unittest.skipUnless(
    torch.cuda.is_available(),
    "context-parallel attention uses CUDA flash-attn (gloo bridges the "
    "collectives, so a single GPU shared by 2 ranks is enough)",
)
@instantiate_parametrized_tests
class TestContextParallelEquivalence(GlooDistributedTestBase):
    """Single-GPU CP attention parity vs a non-causal full-sequence SDPA.

    Two gloo ranks share ``cuda:0``.  Each rank holds a sequence-dim chunk of
    Q/K/V; ``ContextParallelFlashAttention`` all-gathers K/V (over gloo, on CPU)
    and runs flash-attention on the GPU.  The rank's output and Q/K/V gradients
    must match the corresponding chunk of the full-sequence reference.
    """

    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [128, 256, 1024], name_fn=lambda x: f"seq={x}")
    def test_cp_attention_matches_single(self, batch_size: int, seq_len: int):
        from cornstarch.distributed.context_parallel.attention import (
            ContextParallelFlashAttention,
        )

        nheads, dim = 8, 64  # dim=64 / heads=8 are flash-attn-supported shapes
        assert seq_len % self.world_size == 0

        # Full-sequence Q/K/V, identical on every rank (seed reset in _run).
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, nheads, dim),
                device="cuda",
                dtype=DTYPE,
            ).normal_(mean=0, std=0.5),
        )
        for t in (query, key, value):
            t.requires_grad_()

        # Reference: non-causal SDPA over the full sequence. SDPA wants
        # (b, h, s, d); the CP kernel uses (b, s, h, d).
        ref_out = sdpa(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=False,
        ).transpose(1, 2)

        local_q = (
            torch.chunk(query, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_k = (
            torch.chunk(key, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_v = (
            torch.chunk(value, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        # WORLD group as the CP group: avoids a DeviceMesh(device_type="cuda")
        # under the gloo backend (cuda-device-type + gloo-backend mismatch).
        cp_out = ContextParallelFlashAttention.apply(
            local_q, local_k, local_v, dist.GroupMember.WORLD
        )

        torch.testing.assert_close(
            torch.chunk(ref_out, self.world_size, dim=1)[self.rank],
            cp_out,
            atol=CP_ATOL,
            rtol=CP_RTOL,
        )

        # Backward parity: each rank's dq/dk/dv match its chunk of the ref grads.
        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(
            ref_out, [query, key, value], dout
        )

        cp_dout = torch.chunk(dout, self.world_size, dim=1)[self.rank].contiguous()
        cp_dq, cp_dk, cp_dv = torch.autograd.grad(
            cp_out, [local_q, local_k, local_v], cp_dout
        )

        torch.testing.assert_close(
            torch.chunk(ref_dq, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dq,
            atol=CP_ATOL,
            rtol=CP_RTOL,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dk, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dk,
            atol=CP_ATOL,
            rtol=CP_RTOL,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dv, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dv,
            atol=CP_ATOL,
            rtol=CP_RTOL,
        )

    def test_cp_attention_hf_dispatch_wrapper(self):
        """Cover the HF dispatch entry point (the (b, h, s, d) transpose path)."""
        from cornstarch.distributed.context_parallel.attention import (
            context_parallel_flash_attention,
        )

        batch_size, nheads, seq_len, dim = 2, 8, 256, 64
        assert seq_len % self.world_size == 0

        # HF layout is (b, h, s, d); the wrapper transposes to (b, s, h, d).
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, nheads, seq_len, dim),
                device="cuda",
                dtype=DTYPE,
            ).normal_(mean=0, std=0.5),
        )

        ref_out = sdpa(query, key, value, is_causal=False)

        local_q = torch.chunk(query, self.world_size, dim=2)[self.rank].contiguous()
        local_k = torch.chunk(key, self.world_size, dim=2)[self.rank].contiguous()
        local_v = torch.chunk(value, self.world_size, dim=2)[self.rank].contiguous()

        cp_out, _ = context_parallel_flash_attention(
            module=None,
            query=local_q,
            key=local_k,
            value=local_v,
            cp_group=dist.GroupMember.WORLD,
        )

        torch.testing.assert_close(
            torch.chunk(ref_out, self.world_size, dim=2)[self.rank],
            cp_out,
            atol=CP_ATOL,
            rtol=CP_RTOL,
        )


if __name__ == "__main__":
    unittest.main()
