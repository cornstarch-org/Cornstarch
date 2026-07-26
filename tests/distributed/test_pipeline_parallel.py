"""Tests for pipeline parallel P2P communication and 1F1B schedule.

Uses 2 ranks (1 PP stage per rank) with the gloo backend.  Verifies that:
- Rank 0 (stage 0) sends activations and receives gradients.
- Rank 1 (stage 1) receives activations, produces loss, sends gradients.
- The full 1F1B loop accumulates a non-None loss on rank 1 only.
"""
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.pipeline_parallel.p2p import PipelineP2PCommunication


# ---------------------------------------------------------------------------
# Minimal models for PP testing
# ---------------------------------------------------------------------------

class Stage0Model(nn.Module):
    """First stage: linear projection, returns hidden_states dict."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        return {"hidden_states": self.proj(input_ids.float())}


class Stage1Model(nn.Module):
    """Last stage: receives hidden_states and produces a scalar loss."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(4, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.head(hidden_states).mean()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineP2P(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _make_mesh(self) -> ModalProcessGroupMesh:
        return ModalProcessGroupMesh(
            device_type="cpu",
            global_ranks=[0, 1],
            dp_size=1,
            cp_size=1,
            tp_size=1,
            num_pp_stages=2,
        )

    def test_send_recv_tensor(self):
        """Stage 0 sends a tensor; stage 1 receives and verifies."""
        rank = dist.get_rank()
        mesh = self._make_mesh()
        comm = PipelineP2PCommunication(mesh)

        if rank == 0:
            data = {"hidden_states": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
            comm.send_forward(data)
        else:
            received = comm.recv_forward()
            self.assertIsNotNone(received)
            self.assertIn("hidden_states", received)
            self.assertEqual(received["hidden_states"].shape, (1, 4))
            self.assertTrue(
                torch.allclose(received["hidden_states"], torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            )

    def test_send_recv_backward(self):
        """Stage 1 sends a grad; stage 0 receives it."""
        rank = dist.get_rank()
        mesh = self._make_mesh()
        comm = PipelineP2PCommunication(mesh)

        if rank == 1:
            grad = {"hidden_states": torch.tensor([[0.1, 0.2, 0.3, 0.4]])}
            comm.send_backward(grad)
        else:
            received = comm.recv_backward()
            self.assertIsNotNone(received)
            self.assertIn("hidden_states", received)

    def test_1f1b_forward_pass(self):
        """1F1B forward pass on CornstarchLanguageModel via the schedule API."""
        from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import PipelineParallelForwardSpec
        from cornstarch.distributed.pipeline_parallel.schedule import (
            MeshLayout,
            OneForwardOneBackwardSchedule,
        )
        from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture, from_hf_config
        from tests.model.model_configs import llama_config

        mesh = self._make_mesh()

        torch.manual_seed(42)
        config = llama_config()
        config.vocab_size = 128
        config.tie_word_embeddings = False
        model = from_hf_config(config, model_kind="language", attn_implementation="eager")

        total_layers = len(model.decoder_layers)
        start, end = mesh.distribute_layers(total_layers)
        model.decoder_layers = torch.nn.ModuleList(
            list(model.decoder_layers)[start:end]
        )
        model.forward_spec = PipelineParallelForwardSpec(model.forward_spec, mesh)

        model.set_random_init()
        model.materialize("cpu")
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(0.01)
        model.train()

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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        batch = {
            "input_ids": torch.randint(0, 128, (2, 8)),
            "labels": torch.randint(0, 128, (2, 8)),
        }
        microbatches = [{k: v[i : i + 1] for k, v in batch.items()} for i in range(2)]

        def criterion(output, batch):
            if isinstance(output, torch.Tensor):
                return output
            return output.loss if hasattr(output, "loss") else output["loss"]

        result = schedule.step(microbatches, criterion, optimizer, return_loss=True)

        if mesh.is_last_stage():
            self.assertIsNotNone(result["loss"])
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_zb_h2_matches_1f1b_loss_and_gradients(self):
        """ZB-H2 matches 1F1B over the same three microbatches on two ranks."""
        from cornstarch.distributed.pipeline_parallel.deferred_weight_grad import (
            apply_deferred_weight_gradients,
        )
        from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import (
            PipelineParallelForwardSpec,
        )
        from cornstarch.distributed.pipeline_parallel.schedule import MeshLayout
        from cornstarch.distributed.pipeline_parallel.schedule_1f1b import (
            OneForwardOneBackwardSchedule,
        )
        from cornstarch.distributed.pipeline_parallel.schedule_zbpp import (
            ZeroBubblePipelineSchedule,
        )
        from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture, from_hf_config
        from tests.model.model_configs import llama_config

        mesh = self._make_mesh()

        def make_schedule(schedule_type, *, deferred=False):
            torch.manual_seed(123)
            config = llama_config()
            config.vocab_size = 128
            config.tie_word_embeddings = False
            model = from_hf_config(
                config, model_kind="language", attn_implementation="eager"
            )
            start, end = mesh.distribute_layers(len(model.decoder_layers))
            model.decoder_layers = nn.ModuleList(list(model.decoder_layers)[start:end])
            model.forward_spec = PipelineParallelForwardSpec(model.forward_spec, mesh)
            model.set_random_init()
            model.materialize("cpu")
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(0.01)
            if deferred:
                apply_deferred_weight_gradients(model)
            model.train()

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
            schedule = schedule_type(
                plan,
                output_future,
                {id(model): layout},
                {id(model): mesh},
                mesh.dp_size,
            )
            return model, schedule

        torch.manual_seed(321)
        batch = {
            "input_ids": torch.randint(0, 128, (3, 8)),
            "labels": torch.randint(0, 128, (3, 8)),
        }
        microbatches = [
            {key: value[index : index + 1] for key, value in batch.items()}
            for index in range(3)
        ]

        def criterion(output, _batch):
            if isinstance(output, torch.Tensor):
                return output
            return output.loss if hasattr(output, "loss") else output["loss"]

        baseline_model, baseline_schedule = make_schedule(
            OneForwardOneBackwardSchedule
        )
        baseline_result = baseline_schedule.step(microbatches, criterion)
        baseline_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in baseline_model.named_parameters()
            if parameter.grad is not None
        }

        # The model compiles stage graphs lazily. Clear the graph cache so the
        # second instance is traced with its deferred-linear forwards rather
        # than reusing the ordinary-linear graph from the baseline instance.
        torch._dynamo.reset()

        zb_model, zb_schedule = make_schedule(
            ZeroBubblePipelineSchedule, deferred=True
        )
        zb_result = zb_schedule.step(microbatches, criterion)
        zb_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in zb_model.named_parameters()
            if parameter.grad is not None
        }

        self.assertEqual(baseline_grads.keys(), zb_grads.keys())
        for name in baseline_grads:
            self.assertTrue(
                torch.allclose(baseline_grads[name], zb_grads[name], atol=1e-5),
                name,
            )
        if mesh.is_last_stage():
            self.assertTrue(
                torch.allclose(
                    baseline_result["loss"], zb_result["loss"], atol=1e-6
                )
            )

        # Reuse the schedule after clearing caller-owned gradients. Any stale
        # activation, P2P, seam, or deferred-W state makes this second step fail.
        zb_model.zero_grad(set_to_none=True)
        second_result = zb_schedule.step(microbatches, criterion)
        self.assertTrue(any(p.grad is not None for p in zb_model.parameters()))
        if mesh.is_last_stage():
            self.assertTrue(
                torch.allclose(
                    zb_result["loss"], second_result["loss"], atol=1e-6
                )
            )


if __name__ == "__main__":
    unittest.main()
