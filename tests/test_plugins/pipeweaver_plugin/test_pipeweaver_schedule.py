"""Tests for PipeweaverEncoderTrainingPipeweaverScheduler.

Verifies:
  1. 3-phase execution order via forward/backward hooks:
       all encoder forwards → LLM 1F1B → all encoder backwards
  2. Gradient correctness against a sequential baseline.

World layout: PP=4, TP=1, SP=1, DP=1 (4 ranks total).
  - Rank i hosts encoder stage i and LLM stage i.
"""
from __future__ import annotations

import copy
import functools

import torch
import torch.distributed as dist
from colossalai.interface import OptimizerWrapper
from torch import nn
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.pipeweaver_parallel_plugin import (
    PipeweaverEncoderTrainingPipeweaverScheduler,
    PipeweaverPipelineStageManager,
    PipeweaverProcessGroupMesh,
)

from ...distributed_base import GlooDistributedTestBase

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_H = 8  # hidden size


def criterion(x, *args, **kwargs):
    return (x * x).mean()


# ---------------------------------------------------------------------------
# Test model
# ---------------------------------------------------------------------------

class _PipeweaverModel(nn.Module):
    """Each rank has one encoder Linear and one LLM Linear.

    In PP mode (``stage_manager`` is set) only the rank's own layer is
    computed, selected by ``stage_manager.current_mode`` and
    ``stage_manager.stage``.

    In baseline mode (``stage_manager is None``) the full sequential model
    is run: all encoder layers then all LLM layers.
    """

    def __init__(self, pp_size: int) -> None:
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [nn.Linear(_H, _H, bias=False) for _ in range(pp_size)]
        )
        self.llm_layers = nn.ModuleList(
            [nn.Linear(_H, _H, bias=False) for _ in range(pp_size)]
        )
        self.stage_manager: PipeweaverPipelineStageManager | None = None

    def forward(
        self,
        x: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        if self.stage_manager is None:
            # Baseline: full sequential run.
            inp = x
            for enc in self.enc_layers:
                inp = enc(inp)
            for llm in self.llm_layers:
                inp = llm(inp)
            return inp

        sm = self.stage_manager
        stage = sm.stage
        mode = sm.current_mode

        # Encoder first stage gets raw input; all other stages use hidden_states.
        if mode == "encoder" and sm.is_first_stage():
            inp = x
        else:
            inp = hidden_states

        if mode == "encoder":
            out = self.enc_layers[stage](inp)
            return {"hidden_states": out}
        else:
            out = self.llm_layers[stage](inp)
            if sm.is_last_stage():
                return out
            return {"hidden_states": out}


# ---------------------------------------------------------------------------
# Shared PipelineTemplate for PP=4 (4 stages, 1 layer each)
# ---------------------------------------------------------------------------

_ENC_TEMPLATE = PipelineTemplate(
    "enc",
    [["enc_layers.0"], ["enc_layers.1"], ["enc_layers.2"], ["enc_layers.3"]],
)
_LLM_TEMPLATE = PipelineTemplate(
    "llm",
    [["llm_layers.0"], ["llm_layers.1"], ["llm_layers.2"], ["llm_layers.3"]],
)


def _create_stage_manager() -> PipeweaverPipelineStageManager:
    mesh = PipeweaverProcessGroupMesh(_ENC_TEMPLATE, _LLM_TEMPLATE)
    return PipeweaverPipelineStageManager(mesh, mesh.pp_axis)


# ---------------------------------------------------------------------------
# Test: 3-phase forward/backward order
# ---------------------------------------------------------------------------


@instantiate_parametrized_tests
class TestPipeweaverScheduleOrderClass(GlooDistributedTestBase):
    """Verify that the call order follows all-enc-fwd → LLM-1F1B → all-enc-bwd."""

    @property
    def world_size(self) -> int:
        return 4

    @staticmethod
    def _fwd_hook(module, inp, call_list, name):
        call_list.append(f"{name}_fwd")

    @staticmethod
    def _bwd_hook(module, grad_inp, call_list, name):
        call_list.append(f"{name}_bwd")

    @parametrize("num_microbatches", [4, 8], name_fn=lambda x: f"mb={x}")
    def test_forward_backward_order(self, num_microbatches: int) -> None:
        microbatch_size = 1
        pp_size = self.world_size
        stage = self.rank

        model = _PipeweaverModel(pp_size).to("cuda")
        sm = _create_stage_manager()
        model.stage_manager = sm

        schedule = PipeweaverEncoderTrainingPipeweaverScheduler(
            sm, num_microbatches, microbatch_size
        )

        call_list: list[str] = []

        # Register hooks on this rank's enc and llm layers.
        model.enc_layers[stage].register_forward_pre_hook(
            functools.partial(self._fwd_hook, call_list=call_list, name="enc")
        )
        model.llm_layers[stage].register_forward_pre_hook(
            functools.partial(self._fwd_hook, call_list=call_list, name="llm")
        )
        model.enc_layers[stage].register_full_backward_pre_hook(
            functools.partial(self._bwd_hook, call_list=call_list, name="enc")
        )
        model.llm_layers[stage].register_full_backward_pre_hook(
            functools.partial(self._bwd_hook, call_list=call_list, name="llm")
        )

        optimizer = OptimizerWrapper(torch.optim.SGD(model.parameters(), lr=1))
        input_data = torch.rand(num_microbatches * microbatch_size, _H, device="cuda")
        dist.all_reduce(input_data)

        schedule.forward_backward_step(
            model,
            iter([{"x": input_data}]),
            criterion,
            optimizer,
        )

        dist.barrier()
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # Verify 3-phase order:
        #   Phase 1: M × enc_fwd
        #   Phase 2: M × llm_fwd + M × llm_bwd (1F1B interleaved)
        #   Phase 3: M × enc_bwd
        # ------------------------------------------------------------------

        M = num_microbatches
        num_warmup = min(pp_size - stage - 1, M)
        num_remaining = M - num_warmup

        # Build expected call_list for this rank
        expected: list[str] = []
        # Phase 1: all encoder forwards
        expected += ["enc_fwd"] * M
        # Phase 2 warmup: pure LLM forwards
        expected += ["llm_fwd"] * num_warmup
        # Phase 2 steady state: interleaved LLM fwd + bwd
        expected += ["llm_fwd", "llm_bwd"] * num_remaining
        # Phase 2 cooldown: pure LLM backwards
        expected += ["llm_bwd"] * num_warmup
        # Phase 3: all encoder backwards
        expected += ["enc_bwd"] * M

        assert call_list == expected, (
            f"rank {stage}: expected {expected}, got {call_list}"
        )


# ---------------------------------------------------------------------------
# Test: gradient correctness
# ---------------------------------------------------------------------------


@instantiate_parametrized_tests
class TestPipeweaverScheduleGradientClass(GlooDistributedTestBase):
    """Verify PP gradients match sequential baseline for each rank's layers."""

    @property
    def world_size(self) -> int:
        return 4

    @parametrize("num_microbatches", [4, 8, 12], name_fn=lambda x: f"mb={x}")
    @parametrize("microbatch_size", [1, 2], name_fn=lambda x: f"mbs={x}")
    def test_gradient_correctness(
        self, num_microbatches: int, microbatch_size: int
    ) -> None:
        pp_size = self.world_size
        stage = self.rank

        # baseline model — same initialization on every rank (reset_seed).
        model = _PipeweaverModel(pp_size).to("cuda")
        # PP model — deep copy so gradients accumulate separately.
        pp_model = copy.deepcopy(model)

        sm = _create_stage_manager()
        pp_model.stage_manager = sm

        schedule = PipeweaverEncoderTrainingPipeweaverScheduler(
            sm, num_microbatches, microbatch_size
        )

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1)
        pp_optimizer = OptimizerWrapper(
            torch.optim.SGD(pp_model.parameters(), lr=1)
        )

        # Same input on all ranks.
        total_batch = num_microbatches * microbatch_size
        input_data = torch.rand(total_batch, _H, device="cuda")
        dist.all_reduce(input_data)

        # ------------------------------------------------------------------
        # Baseline: full sequential forward + backward on all ranks.
        # ------------------------------------------------------------------
        output = model(x=input_data)
        loss = criterion(output)
        loss.backward()

        # ------------------------------------------------------------------
        # PP schedule.
        # ------------------------------------------------------------------
        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter([{"x": input_data}]),
            criterion,
            pp_optimizer,
            return_loss=True,
        )

        dist.barrier()
        torch.cuda.synchronize()

        # Loss check on LLM last stage.
        if sm.is_last_stage():
            torch.testing.assert_close(
                loss,
                pp_ret["loss"],
                atol=5e-3,
                rtol=5e-3,
                msg=f"rank {stage}: loss mismatch",
            )

        # Gradient check: each rank compares its own stage's layers.
        torch.testing.assert_close(
            model.enc_layers[stage].weight.grad,
            pp_model.enc_layers[stage].weight.grad,
            atol=5e-3,
            rtol=5e-3,
            msg=f"rank {stage}: enc_layers[{stage}].weight.grad mismatch",
        )
        torch.testing.assert_close(
            model.llm_layers[stage].weight.grad,
            pp_model.llm_layers[stage].weight.grad,
            atol=5e-3,
            rtol=5e-3,
            msg=f"rank {stage}: llm_layers[{stage}].weight.grad mismatch",
        )

    @parametrize("num_microbatches", [4, 8], name_fn=lambda x: f"mb={x}")
    def test_optimizer_step_correctness(self, num_microbatches: int) -> None:
        """Verify that weights are updated identically after optimizer.step()."""
        pp_size = self.world_size
        stage = self.rank
        microbatch_size = 1

        model = _PipeweaverModel(pp_size).to("cuda")
        pp_model = copy.deepcopy(model)

        sm = _create_stage_manager()
        pp_model.stage_manager = sm

        schedule = PipeweaverEncoderTrainingPipeweaverScheduler(
            sm, num_microbatches, microbatch_size
        )

        lr = 0.01
        model_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=lr))

        total_batch = num_microbatches * microbatch_size
        input_data = torch.rand(total_batch, _H, device="cuda")
        dist.all_reduce(input_data)

        # Baseline
        output = model(x=input_data)
        loss = criterion(output)
        loss.backward()
        model_optimizer.step()
        model_optimizer.zero_grad()

        # PP
        schedule.forward_backward_step(
            pp_model,
            iter([{"x": input_data}]),
            criterion,
            pp_optimizer,
        )
        pp_optimizer.step()
        pp_optimizer.zero_grad()

        dist.barrier()

        # Check updated weights for this rank's stage layers.
        torch.testing.assert_close(
            model.enc_layers[stage].weight,
            pp_model.enc_layers[stage].weight,
            atol=5e-3,
            rtol=5e-3,
            msg=f"rank {stage}: enc_layers[{stage}].weight mismatch after step",
        )
        torch.testing.assert_close(
            model.llm_layers[stage].weight,
            pp_model.llm_layers[stage].weight,
            atol=5e-3,
            rtol=5e-3,
            msg=f"rank {stage}: llm_layers[{stage}].weight mismatch after step",
        )
