from __future__ import annotations

import copy
import functools
from typing import Any

import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from torch import nn
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
    MultiModalPipelineStageManager,
    MultiModalProcessGroupMesh,
)

from ...distributed_base import GlooDistributedTestBase

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_H = 8  # hidden size used by all heterogeneous test models


class _HeteroModel(nn.Module):
    """
    Simple position-independent model (linear layers only, no cross-token
    interaction) shared by both heterogeneous test classes.

    Architecture:
      - enc  : Linear(H, H, bias=False)  — used by the encoder PP stage
      - llm0 : Linear(H, H, bias=False)  — used by LLM PP stage 0
      - llm1 : Linear(H, H, bias=False)  — used by LLM PP stage 1

    When pp_config is None the full model runs sequentially on rank 0
    (used as a non-PP baseline).  When pp_config is set, each rank
    executes only its assigned layer(s).

    The model uses ``hidden_states`` as the P2P key so that the schedule's
    ``_merge_cross_modal_recv`` / ``_split_cross_modal_grad`` paths are
    exercised end-to-end.
    """

    def __init__(self, enc_modal_name: str, llm_modal_name: str):
        super().__init__()
        self.enc = nn.Linear(_H, _H, bias=False)
        self.llm0 = nn.Linear(_H, _H, bias=False)
        self.llm1 = nn.Linear(_H, _H, bias=False)
        self.pp_config: dict | None = None
        self._enc_modal_name = enc_modal_name
        self._llm_modal_name = llm_modal_name

    def forward(
        self,
        x: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        if self.pp_config is None:
            # Non-PP baseline: run full model sequentially.
            return self.llm1(self.llm0(self.enc(x)))

        modal_name: str = self.pp_config["modal_name"]
        sm: MultiModalPipelineStageManager = self.pp_config["stage_manager"]

        if modal_name == self._enc_modal_name:
            out = self.enc(x)
            return {"hidden_states": out}
        else:
            inp = hidden_states if hidden_states is not None else x
            out = self.llm0(inp) if sm.stage_in_modal == 0 else self.llm1(inp)
            if sm.is_last_stage(check_only_in_modal=False):
                return out
            return {"hidden_states": out}


class _HeteroVarModel(nn.Module):
    """Variant of ``_HeteroModel`` with a variable-depth LLM (``nn.ModuleList``).

    Used by ``TestScheduleHeterogeneousTPSPClass`` where ``llm_pp`` is a
    test parameter, so the number of LLM stages is not fixed at 2.

    The encoder is always 1 PP stage.  The LLM has ``llm_num_stages`` PP
    stages, each holding one ``nn.Linear(_H, _H, bias=False)`` layer.

    Architecture in PP mode: each rank runs only its own layer.
    Architecture in baseline mode (pp_config=None): full sequential pass.
    """

    def __init__(self, enc_modal_name: str, llm_modal_name: str, llm_num_stages: int):
        super().__init__()
        self.enc = nn.Linear(_H, _H, bias=False)
        self.llm_layers = nn.ModuleList(
            [nn.Linear(_H, _H, bias=False) for _ in range(llm_num_stages)]
        )
        self.pp_config: dict | None = None
        self._enc_modal_name = enc_modal_name
        self._llm_modal_name = llm_modal_name

    def forward(
        self,
        x: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        if self.pp_config is None:
            out = self.enc(x)
            for layer in self.llm_layers:
                out = layer(out)
            return out

        modal_name: str = self.pp_config["modal_name"]
        sm: MultiModalPipelineStageManager = self.pp_config["stage_manager"]

        if modal_name == self._enc_modal_name:
            return {"hidden_states": self.enc(x)}
        else:
            inp = hidden_states if hidden_states is not None else x
            out = self.llm_layers[sm.stage_in_modal](inp)
            if sm.is_last_stage(check_only_in_modal=False):
                return out
            return {"hidden_states": out}


def create_data() -> list[Any]:
    tensor = torch.ones(1, device=get_accelerator().get_current_device())
    return [
        "tensor",
        tensor,
        [tensor],
        {"tensor": tensor},
    ]


class SingleEncoderModelTestCaseBase:
    @property
    def world_size(self):
        return 4

    class SingleEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
            self.llm = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
            self.pp_config: dict = None

        def get_templates(self) -> list[PipelineTemplate]:
            return [
                PipelineTemplate("encoder", [["encoder.0", "encoder.1", "encoder.2"]]),
                PipelineTemplate(
                    "llm",
                    [["llm.0", "llm.1"], ["llm.2", "llm.3"], ["llm.4", "llm.5"]],
                ),
            ]

        def forward(
            self, x: torch.Tensor = None, input_obj: torch.Tensor = None, **kwargs
        ):
            if self.pp_config:
                modal_name: str = self.pp_config["modal_name"]
                stage_manager: MultiModalPipelineStageManager = self.pp_config[
                    "stage_manager"
                ]
                start_idx, end_idx = self.pp_config["index"]

                if not stage_manager.is_first_stage(check_only_in_modal=False):
                    x = input_obj

                if modal_name == "encoder":
                    assert start_idx >= 0 and end_idx <= 3
                    for layer in self.encoder[start_idx:end_idx]:
                        x = layer(x)
                elif modal_name == "llm":
                    assert start_idx >= 0 and end_idx <= 6
                    for layer in self.llm[start_idx:end_idx]:
                        x = layer(x)

                if stage_manager.is_last_stage(check_only_in_modal=False):
                    return x
                else:
                    return {"input_obj": x}
            else:
                for layer in self.encoder:
                    x = layer(x)
                for layer in self.llm:
                    x = layer(x)

                return x

    def create_stage_manager(
        self, model: SingleEncoderModel
    ) -> MultiModalPipelineStageManager:
        encoder_template, llm_template = model.get_templates()
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={encoder_template: 1},
            llm_template=(llm_template, 1, 1),
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)


def criterion(x, *args, **kwargs):
    return (x * x).mean()


class ForwardBackwardOrderClassBase(GlooDistributedTestBase):
    @staticmethod
    def forward_pre_hook(module: nn.Module, input: torch.Tensor, call_list: list[str]):
        call_list.append("forward")

    @staticmethod
    def backward_pre_hook(
        module: nn.Module, grad_input: torch.Tensor, call_list: list[str]
    ):
        call_list.append("backward")

    def register_hooks(self, module: nn.Module, call_list: list[str]):
        module.register_forward_pre_hook(
            functools.partial(
                ForwardBackwardOrderClassBase.forward_pre_hook,
                call_list=call_list,
            )
        )
        module.register_full_backward_pre_hook(
            functools.partial(
                ForwardBackwardOrderClassBase.backward_pre_hook,
                call_list=call_list,
            )
        )


@instantiate_parametrized_tests
class TestSingleEncoderForwardBackwardOrderClass(
    ForwardBackwardOrderClassBase, SingleEncoderModelTestCaseBase
):

    @property
    def world_size(self) -> int:
        return 4

    @parametrize("num_microbatches", [4, 8, 12], name_fn=lambda x: f"mb={x}")
    def test_forward_backward_order(self, num_microbatches: int):
        microbatch_size = 1
        call_list: list[str] = []

        model = self.SingleEncoderModel().to("cuda")
        model.train(mode=True)
        optimizer = OptimizerWrapper(torch.optim.SGD(model.parameters(), lr=1))

        stage_manager = self.create_stage_manager(model=model)
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager, num_microbatches, microbatch_size
        )

        start_idx, end_idx = stage_manager.get_stage_index(
            stage_manager.distribute_layers()
        )

        my_modal = next(
            template
            for template, ranks in stage_manager.pg_mesh.modal_to_ranks.items()
            if dist.get_rank() in ranks
        )
        assert my_modal.model_name in ["encoder", "llm"]
        self.register_hooks(
            (
                model.encoder[start_idx]
                if my_modal.model_name == "encoder"
                else model.llm[start_idx]
            ),
            call_list,
        )

        model.pp_config = {
            "modal_name": my_modal.model_name,
            "index": (start_idx, end_idx),
            "stage_manager": stage_manager,
        }

        input_list = [torch.rand(num_microbatches * microbatch_size, 8, device="cuda")]

        schedule.forward_backward_step(
            model,
            iter(input_list),
            criterion,
            optimizer,
        )

        dist.barrier()
        torch.cuda.synchronize()

        assert len(call_list) == num_microbatches * 2
        expected_order = (
            ["forward"] * (4 - self.rank)
            + ["backward", "forward"] * (num_microbatches - 4 + self.rank)
            + ["backward"] * (4 - self.rank)
        )

        assert call_list == expected_order


@instantiate_parametrized_tests
class TestScheduleSingleEncoderClass(
    SingleEncoderModelTestCaseBase, GlooDistributedTestBase
):
    @property
    def world_size(self) -> int:
        return 4

    @parametrize("num_microbatches", [4, 8, 12], name_fn=lambda x: f"mb={x}")
    @parametrize("microbatch_size", [1, 2, 4], name_fn=lambda x: f"mbs={x}")
    def test_schedule(self, num_microbatches: int, microbatch_size: int):
        model = self.SingleEncoderModel().to("cuda")
        pp_model = copy.deepcopy(model)

        stage_manager = self.create_stage_manager(model=model)
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager, num_microbatches, microbatch_size
        )

        start_idx, end_idx = stage_manager.get_stage_index(
            stage_manager.distribute_layers()
        )

        my_modal = next(
            template
            for template, ranks in stage_manager.pg_mesh.modal_to_ranks.items()
            if dist.get_rank() in ranks
        )

        assert my_modal.model_name in ["encoder", "llm"]
        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "index": (start_idx, end_idx),
            "stage_manager": stage_manager,
        }

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1)
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=1))

        input_list = [torch.rand(num_microbatches * microbatch_size, 8).to("cuda")]
        dist.all_reduce(input_list[0])

        # forward and backward
        output = model(input_list[0])
        loss = criterion(output)
        loss.backward()

        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter(input_list),
            criterion,
            pp_optimizer,
            return_loss=True,
            return_outputs=True,
        )

        if stage_manager.is_last_stage(check_only_in_modal=False):
            torch.testing.assert_close(loss, pp_ret["loss"], atol=5e-3, rtol=5e-3)

        # check gradients
        if my_modal.model_name == "encoder":
            for i in range(start_idx, end_idx):
                torch.testing.assert_close(
                    model.encoder[i].weight.grad,
                    pp_model.encoder[i].weight.grad,
                    atol=5e-3,
                    rtol=5e-3,
                )
                torch.testing.assert_close(
                    model.encoder[i].bias.grad,
                    pp_model.encoder[i].bias.grad,
                    atol=5e-3,
                    rtol=5e-3,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                torch.testing.assert_close(
                    model.llm[i].weight.grad,
                    pp_model.llm[i].weight.grad,
                    atol=5e-3,
                    rtol=5e-3,
                )
                torch.testing.assert_close(
                    model.llm[i].bias.grad,
                    pp_model.llm[i].bias.grad,
                    atol=5e-3,
                    rtol=5e-3,
                )

        # step
        model_optimizer.step()
        pp_optimizer.step()
        model_optimizer.zero_grad()
        pp_optimizer.zero_grad()

        # check updated param
        if my_modal.model_name == "encoder":
            for i in range(start_idx, end_idx):
                torch.testing.assert_close(
                    model.encoder[i].weight,
                    pp_model.encoder[i].weight,
                    atol=5e-3,
                    rtol=5e-3,
                )
                torch.testing.assert_close(
                    model.encoder[i].bias,
                    pp_model.encoder[i].bias,
                    atol=5e-3,
                    rtol=5e-3,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                torch.testing.assert_close(
                    model.llm[i].weight,
                    pp_model.llm[i].weight,
                    atol=5e-3,
                    rtol=5e-3,
                )
                torch.testing.assert_close(
                    model.llm[i].bias,
                    pp_model.llm[i].bias,
                    atol=5e-3,
                    rtol=5e-3,
                )


# ---------------------------------------------------------------------------
# Heterogeneous SP test  (enc_sp=2, all_to_all mode)
# ---------------------------------------------------------------------------


@instantiate_parametrized_tests
class TestScheduleHeterogeneousSPClass(GlooDistributedTestBase):
    """Schedule-level SP merge/split correctness for enc_sp=2, all_to_all mode.

    World layout (4 ranks):
      rank 0 — encoder SP coord 0  (enc_pp=1, enc_tp=1, enc_sp=2)
      rank 1 — encoder SP coord 1
      rank 2 — LLM PP stage 0     (llm_pp=2, llm_tp=1, llm_sp=1)
      rank 3 — LLM PP stage 1     (last stage, computes loss)

    Correctness criterion
    ---------------------
    Because the model contains only position-independent linear layers and the
    loss is ``(x**2).mean()``, the PP result must exactly match a non-PP
    baseline that runs the full model on the complete 8-token sequence:

      • LLM last stage: ``pp_loss == baseline_loss``
      • Encoder ranks (each holds half the sequence gradient):
        all_reduce(pp_model.enc.weight.grad) == baseline enc.weight.grad
      • LLM stage 0/1 ranks: direct gradient equality with baseline
    """

    _ENC = PipelineTemplate("enc_sp2", [["enc"]])
    _LLM = PipelineTemplate("llm_2pp", [["llm0"], ["llm1"]])

    @property
    def world_size(self) -> int:
        return 4

    def _make_model(self) -> _HeteroModel:
        return _HeteroModel(
            enc_modal_name=self._ENC.model_name,
            llm_modal_name=self._LLM.model_name,
        ).to("cuda")

    def _create_stage_manager(self) -> MultiModalPipelineStageManager:
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={self._ENC: (1, 2)},  # tp=1, sp=2
            llm_template=(self._LLM, 1, 1),  # tp=1, sp=1
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    @parametrize("num_microbatches", [2, 4], name_fn=lambda x: f"mb={x}")
    def test_sp_merge_and_grad_split(self, num_microbatches: int):
        total_seq = 8  # full sequence length across both encoder SP ranks
        seq_per_sp = total_seq // 2  # tokens owned by each encoder SP rank
        microbatch_size = seq_per_sp // num_microbatches

        # Identical initial weights on all ranks.
        model_ref = self._make_model()
        for p in model_ref.parameters():
            dist.broadcast(p.data, src=0)
        pp_model = copy.deepcopy(model_ref)

        stage_manager = self._create_stage_manager()
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager,
            num_microbatches=num_microbatches,
            microbatch_size=microbatch_size,
            encoder_sp_gather=False,
        )

        # Identical full input on all ranks.
        x_full = torch.rand(total_seq, _H, device="cuda")
        dist.broadcast(x_full, src=0)

        # Non-PP baseline (computed on every rank; results are identical because
        # weights and input are the same everywhere).
        y_ref = model_ref.enc(x_full)
        loss_ref = criterion(model_ref.llm1(model_ref.llm0(y_ref)))
        loss_ref.backward()

        # PP schedule — each encoder SP rank provides its own sequence slice.
        pg_mesh = stage_manager.pg_mesh
        my_modal = pg_mesh.my_modal
        sp_coord = pg_mesh.coords[0][pg_mesh.sp_axis]

        if my_modal.model_name == self._ENC.model_name:
            x_local = x_full[sp_coord * seq_per_sp : (sp_coord + 1) * seq_per_sp]
            input_list = [x_local.detach().clone()]
        else:
            # LLM ranks: micro_batch is unused; hidden_states arrive via P2P.
            input_list = [
                torch.zeros(num_microbatches * microbatch_size, _H, device="cuda")
            ]

        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "stage_manager": stage_manager,
        }
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=0))
        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter(input_list),
            criterion,
            pp_optimizer,
            return_loss=True,
        )

        dist.barrier()

        # Loss check on the globally last stage.
        if stage_manager.is_last_stage(check_only_in_modal=False):
            torch.testing.assert_close(loss_ref, pp_ret["loss"], atol=1e-4, rtol=1e-4)

        # Gradient checks.
        # get_group_along_axis is collective; all ranks must call it.
        enc_sp_group = pg_mesh.get_group_along_axis(pg_mesh.sp_axis)

        if my_modal.model_name == self._ENC.model_name:
            # Each encoder SP rank holds the contribution of its sequence slice.
            # After all-reducing across SP ranks the result equals the baseline
            # full-sequence gradient (linear decomposition of the gradient sum).
            enc_grad = pp_model.enc.weight.grad.clone()
            dist.all_reduce(enc_grad, group=enc_sp_group)
            torch.testing.assert_close(
                model_ref.enc.weight.grad, enc_grad, atol=1e-4, rtol=1e-4
            )
        else:
            # LLM stages process the fully merged sequence — same as baseline.
            if stage_manager.stage_in_modal == 0:
                torch.testing.assert_close(
                    model_ref.llm0.weight.grad,
                    pp_model.llm0.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )
            else:
                torch.testing.assert_close(
                    model_ref.llm1.weight.grad,
                    pp_model.llm1.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )


# ---------------------------------------------------------------------------
# Heterogeneous TP test  (enc_tp=2, LLM tp=1 — TP fan-in / deduplication)
# ---------------------------------------------------------------------------


@instantiate_parametrized_tests
class TestScheduleHeterogeneousTPClass(GlooDistributedTestBase):
    """Schedule-level TP deduplication correctness for enc_tp=2, llm_tp=1.

    World layout (4 ranks):
      rank 0 — encoder TP coord 0  (enc_pp=1, enc_tp=2, enc_sp=1)
      rank 1 — encoder TP coord 1
      rank 2 — LLM PP stage 0     (llm_pp=2, llm_tp=1, llm_sp=1)
      rank 3 — LLM PP stage 1     (last stage, computes loss)

    Both encoder TP ranks receive the same input and start with identical
    weights, so they produce identical outputs (simulating an all-reduce
    inside the real encoder).  ``_merge_cross_modal_recv`` deduplicates by
    taking the first of the two received tensors (tp_group_size=2).  On the
    backward pass ``_split_cross_modal_grad`` broadcasts the same gradient
    to both encoder TP ranks.

    Correctness criterion
    ---------------------
      • LLM last stage: ``pp_loss == baseline_loss``
      • Each encoder TP rank: ``pp_model.enc.weight.grad == baseline``
        (both ranks compute the same gradient because they ran the same
         forward pass with the same input and receive the same gradient)
      • LLM stage 0/1 ranks: direct gradient equality with baseline
    """

    _ENC = PipelineTemplate("enc_tp2", [["enc"]])
    _LLM = PipelineTemplate("llm_2pp_b", [["llm0"], ["llm1"]])

    @property
    def world_size(self) -> int:
        return 4

    def _make_model(self) -> _HeteroModel:
        return _HeteroModel(
            enc_modal_name=self._ENC.model_name,
            llm_modal_name=self._LLM.model_name,
        ).to("cuda")

    def _create_stage_manager(self) -> MultiModalPipelineStageManager:
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={self._ENC: (2, 1)},  # tp=2, sp=1
            llm_template=(self._LLM, 1, 1),  # tp=1, sp=1
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    @parametrize("num_microbatches", [2, 4], name_fn=lambda x: f"mb={x}")
    def test_tp_dedup_and_grad_broadcast(self, num_microbatches: int):
        total_seq = 8
        microbatch_size = total_seq // num_microbatches

        # Identical initial weights on all ranks.
        model_ref = self._make_model()
        for p in model_ref.parameters():
            dist.broadcast(p.data, src=0)
        pp_model = copy.deepcopy(model_ref)

        stage_manager = self._create_stage_manager()
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager,
            num_microbatches=num_microbatches,
            microbatch_size=microbatch_size,
        )

        # Identical full input on all ranks.
        x_full = torch.rand(total_seq, _H, device="cuda")
        dist.broadcast(x_full, src=0)

        # Non-PP baseline.
        y_ref = model_ref.enc(x_full)
        loss_ref = criterion(model_ref.llm1(model_ref.llm0(y_ref)))
        loss_ref.backward()

        # PP schedule — both encoder TP ranks process the same full input,
        # producing identical outputs (all-reduce already happened inside enc).
        pg_mesh = stage_manager.pg_mesh
        my_modal = pg_mesh.my_modal

        if my_modal.model_name == self._ENC.model_name:
            input_list = [x_full.detach().clone()]
        else:
            input_list = [
                torch.zeros(num_microbatches * microbatch_size, _H, device="cuda")
            ]

        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "stage_manager": stage_manager,
        }
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=0))
        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter(input_list),
            criterion,
            pp_optimizer,
            return_loss=True,
        )

        dist.barrier()

        # Loss check on the globally last stage.
        if stage_manager.is_last_stage(check_only_in_modal=False):
            torch.testing.assert_close(loss_ref, pp_ret["loss"], atol=1e-4, rtol=1e-4)

        if my_modal.model_name == self._ENC.model_name:
            # Both TP ranks ran identical forward passes and received the same
            # gradient → their enc.weight.grad must equal the baseline.
            torch.testing.assert_close(
                model_ref.enc.weight.grad,
                pp_model.enc.weight.grad,
                atol=1e-4,
                rtol=1e-4,
            )
        else:
            if stage_manager.stage_in_modal == 0:
                torch.testing.assert_close(
                    model_ref.llm0.weight.grad,
                    pp_model.llm0.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )
            else:
                torch.testing.assert_close(
                    model_ref.llm1.weight.grad,
                    pp_model.llm1.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )


# ---------------------------------------------------------------------------
# Combined TP+SP test — parametrized over (enc_tp, enc_sp, llm_tp, llm_sp)
# ---------------------------------------------------------------------------

# Nine configs, all world_size=8 (dp=1, enc_pp=1, llm_pp=1):
#   enc_tp * enc_sp  +  llm_tp * llm_sp  = 8
#
# enc_tp enc_sp  llm_tp llm_sp  tp_group_size  sp_A  scenario
# ──────────────────────────────────────────────────────────────────────────
#   2      2       4      1          1           2    TP fan-out + SP concat(2), llm_sp=1
#   1      4       4      1          1           4    TP fan-out + SP concat(4), llm_sp=1
#   4      1       4      1          1           1    equal TP(4), no concat,    llm_sp=1
#   2      2       2      2          1           2    equal TP + SP concat(2),   llm_sp=2
#   1      4       2      2          1           4    SP concat(4),              llm_sp=2
#   4      1       2      2          2           1    TP dedup(2),               llm_sp=2
#   2      2       1      4          2           2    TP dedup(2) + SP concat(2),llm_sp=4
#   1      4       1      4          1           4    SP concat(4),              llm_sp=4
#   4      1       1      4          4           1    TP dedup(4),               llm_sp=4
_TPSP_CONFIGS = [
    # llm_sp=1
    (2, 2, 4, 1),
    (1, 4, 4, 1),
    (4, 1, 4, 1),
    # llm_sp=2
    (2, 2, 2, 2),
    (1, 4, 2, 2),
    (4, 1, 2, 2),
    # llm_sp=4
    (2, 2, 1, 4),
    (1, 4, 1, 4),
    (4, 1, 1, 4),
]


@instantiate_parametrized_tests
class TestScheduleHeterogeneousTPSPClass(GlooDistributedTestBase):
    """Schedule-level correctness for the combined TP + SP border path.

    The focus is the encoder→LLM border: how tensors are merged on the LLM
    first stage (``_merge_cross_modal_recv``) and how gradients are split back
    to encoder ranks (``_split_cross_modal_grad``).

    LLM pipeline depth is fixed at 1 (``llm_pp=1``) so the test is not
    diluted by LLM PP complexity.  All 9 configurations use exactly 8 ranks
    (dp=1, enc_pp=1):  enc_tp * enc_sp + llm_tp * llm_sp = 8.

    The 3×3 grid covers:
      • llm_sp ∈ {1, 2, 4}   — LLM sequence parallelism
      • enc_sp ∈ {2, 4, 1}   — encoder sequence parallelism (sp_A ∈ {2, 4, 1})
      • enc_tp ∈ {2, 1, 4}   — encoder tensor parallelism (dedup levels)
      • llm_tp ∈ {4, 2, 1}   — LLM tensor parallelism (fan-out/equal/fan-in)

    Scenarios covered: TP fan-out, equal TP, TP dedup (2× and 4×), SP concat
    (2-way and 4-way), and all combinations thereof.

    Correctness criterion
    ---------------------
    Every LLM rank holds the full (non-sharded) weight, so all LLM TP/SP
    ranks compute identical outputs and gradients.  This means:

      • All LLM ranks (is_last_stage globally): pp_loss == baseline_loss

      • Encoder ranks — two assertions:
        (a) TP consistency within the same SP group:
              all_reduce(grad, tp_axis_group) == enc_tp * grad
        (b) Combined SP + TP accuracy:
              all_reduce(grad, all_enc_group) / enc_tp == baseline

      • Each LLM rank: pp_model.llm_layers[0].weight.grad == baseline

    Schedule parameters: total_seq=16, num_microbatches ∈ {2, 4}.
    With microbatch_size = (total_seq // enc_sp) // num_microbatches ≥ 1 for
    all configurations.
    """

    @property
    def world_size(self) -> int:
        return 8

    def _make_model(self, enc_modal_name: str, llm_modal_name: str):
        return _HeteroVarModel(
            enc_modal_name=enc_modal_name,
            llm_modal_name=llm_modal_name,
            llm_num_stages=1,
        ).to("cuda")

    def _create_stage_manager(
        self,
        enc_tp: int,
        enc_sp: int,
        llm_tp: int,
        llm_sp: int,
        enc_modal_name: str,
        llm_modal_name: str,
    ) -> MultiModalPipelineStageManager:
        enc_template = PipelineTemplate(enc_modal_name, [["enc"]])
        llm_template = PipelineTemplate(llm_modal_name, [["llm_layers.0"]])
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={enc_template: (enc_tp, enc_sp)},
            llm_template=(llm_template, llm_tp, llm_sp),
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    @parametrize(
        "enc_tp,enc_sp,llm_tp,llm_sp",
        _TPSP_CONFIGS,
        name_fn=lambda e_tp, e_sp, l_tp, l_sp: (
            f"enc_tp{e_tp}sp{e_sp}__llm_tp{l_tp}sp{l_sp}"
        ),
    )
    @parametrize("num_microbatches", [2, 4], name_fn=lambda x: f"mb={x}")
    def test_combined_tp_sp(
        self,
        enc_tp: int,
        enc_sp: int,
        llm_tp: int,
        llm_sp: int,
        num_microbatches: int,
    ):
        total_seq = 16
        seq_per_sp = total_seq // enc_sp
        microbatch_size = seq_per_sp // num_microbatches

        enc_modal_name = f"enc_tp{enc_tp}sp{enc_sp}"
        llm_modal_name = f"llm_tp{llm_tp}sp{llm_sp}"

        model_ref = self._make_model(enc_modal_name, llm_modal_name)
        for p in model_ref.parameters():
            dist.broadcast(p.data, src=0)
        pp_model = copy.deepcopy(model_ref)

        stage_manager = self._create_stage_manager(
            enc_tp, enc_sp, llm_tp, llm_sp, enc_modal_name, llm_modal_name
        )
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager,
            num_microbatches=num_microbatches,
            microbatch_size=microbatch_size,
            encoder_sp_gather=False,
        )

        # Full input shared across all ranks.
        x_full = torch.rand(total_seq, _H, device="cuda")
        dist.broadcast(x_full, src=0)

        # Non-PP baseline: same weights and full input on every rank.
        # The loss function (x*x).mean() decomposes linearly across microbatches,
        # so pp_ret["loss"] (sum of per-microbatch losses / num_microbatches) equals
        # this scalar exactly.
        y_ref = model_ref.enc(x_full)
        out_ref = model_ref.llm_layers[0](y_ref)
        loss_ref = criterion(out_ref)
        loss_ref.backward()

        # PP schedule.
        # Each encoder SP group processes its contiguous sequence slice.
        # All TP ranks in the same SP group process the same slice (simulating
        # the all-reduce that collapses TP redundancy in the real encoder).
        pg_mesh = stage_manager.pg_mesh
        my_modal = pg_mesh.my_modal
        sp_coord = pg_mesh.coords[0][pg_mesh.sp_axis]

        if my_modal.model_name == enc_modal_name:
            x_local = x_full[sp_coord * seq_per_sp : (sp_coord + 1) * seq_per_sp]
            input_list = [x_local.detach().clone()]
        else:
            input_list = [
                torch.zeros(num_microbatches * microbatch_size, _H, device="cuda")
            ]

        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "stage_manager": stage_manager,
        }
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=0))
        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter(input_list),
            criterion,
            pp_optimizer,
            return_loss=True,
        )

        dist.barrier()

        # --- Loss check on all LLM ranks (all are globally last with llm_pp=1) ---
        if stage_manager.is_last_stage(check_only_in_modal=False):
            torch.testing.assert_close(loss_ref, pp_ret["loss"], atol=1e-4, rtol=1e-4)

        # --- Gradient checks (get_group_along_axis is collective: all 8 ranks call it) ---
        tp_group = pg_mesh.get_group_along_axis(pg_mesh.tp_axis)
        enc_all_group = pg_mesh.get_group_along_axis([pg_mesh.sp_axis, pg_mesh.tp_axis])

        if my_modal.model_name == enc_modal_name:
            enc_grad = pp_model.enc.weight.grad.clone()

            # (a) TP consistency: all enc TP ranks in the same SP group processed
            # the same sequence slice and received the same gradient chunk, so
            # their gradients are identical.  all_reduce == enc_tp * individual.
            tp_sum = enc_grad.clone()
            dist.all_reduce(tp_sum, group=tp_group)
            torch.testing.assert_close(tp_sum, enc_grad * enc_tp, atol=1e-4, rtol=1e-4)

            # (b) Combined SP + TP accuracy.
            # Summing all encoder gradients gives enc_tp copies of the baseline
            # (one per TP rank per SP group), so sum / enc_tp == baseline.
            full_sum = enc_grad.clone()
            dist.all_reduce(full_sum, group=enc_all_group)
            full_sum /= enc_tp
            torch.testing.assert_close(
                model_ref.enc.weight.grad, full_sum, atol=1e-4, rtol=1e-4
            )
        else:
            # All LLM TP/SP ranks receive the full merged sequence and compute the
            # same gradient as the non-PP baseline.
            torch.testing.assert_close(
                model_ref.llm_layers[0].weight.grad,
                pp_model.llm_layers[0].weight.grad,
                atol=1e-4,
                rtol=1e-4,
            )


# ---------------------------------------------------------------------------
# ring_attn / encoder_sp_gather=True test
# ---------------------------------------------------------------------------


@instantiate_parametrized_tests
class TestScheduleEncoderSPGatherClass(GlooDistributedTestBase):
    """Schedule-level correctness for the ring_attn (encoder_sp_gather=True) path.

    World layout (4 ranks):
      rank 0 — encoder SP coord 0  (enc_pp=1, enc_tp=1, enc_sp=2)
      rank 1 — encoder SP coord 1
      rank 2 — LLM PP stage 0     (llm_pp=2, llm_tp=1, llm_sp=1)
      rank 3 — LLM PP stage 1     (last stage, computes loss)

    In ring_attn mode the projector inside the encoder calls
    ``gather_forward_split_backward`` before sending, so every encoder SP
    rank holds the FULL sequence output.  The schedule uses
    ``encoder_sp_gather=True``, which means:
      - Forward: ``_merge_cross_modal_recv`` takes the first of sp_A identical
        full-sequence tensors (no concatenation).
      - Backward: ``_split_cross_modal_grad`` broadcasts the same full gradient
        to all encoder SP ranks (no splitting).

    To simulate this in the test, all encoder SP ranks receive the complete
    input and produce identical outputs.

    Correctness criterion
    ---------------------
      • LLM last stage: pp_loss == baseline_loss
      • Each encoder SP rank independently: pp_model.enc.weight.grad == baseline
        (no all-reduce needed — both ranks ran the same forward and got the full grad)
      • LLM stage 0/1 ranks: direct gradient equality with baseline
    """

    _ENC = PipelineTemplate("enc_sp2_gather", [["enc"]])
    _LLM = PipelineTemplate("llm_2pp_d", [["llm0"], ["llm1"]])

    @property
    def world_size(self) -> int:
        return 4

    def _make_model(self) -> _HeteroModel:
        return _HeteroModel(
            enc_modal_name=self._ENC.model_name,
            llm_modal_name=self._LLM.model_name,
        ).to("cuda")

    def _create_stage_manager(self) -> MultiModalPipelineStageManager:
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={self._ENC: (1, 2)},  # tp=1, sp=2
            llm_template=(self._LLM, 1, 1),  # tp=1, sp=1
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    @parametrize("num_microbatches", [2, 4], name_fn=lambda x: f"mb={x}")
    def test_sp_gather_take_first(self, num_microbatches: int):
        total_seq = 8
        microbatch_size = total_seq // num_microbatches

        model_ref = self._make_model()
        for p in model_ref.parameters():
            dist.broadcast(p.data, src=0)
        pp_model = copy.deepcopy(model_ref)

        stage_manager = self._create_stage_manager()
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager,
            num_microbatches=num_microbatches,
            microbatch_size=microbatch_size,
            encoder_sp_gather=True,  # ring_attn: each encoder SP rank holds the full sequence
        )

        x_full = torch.rand(total_seq, _H, device="cuda")
        dist.broadcast(x_full, src=0)

        # Non-PP baseline.
        y_ref = model_ref.enc(x_full)
        loss_ref = criterion(model_ref.llm1(model_ref.llm0(y_ref)))
        loss_ref.backward()

        # PP schedule.
        # All encoder SP ranks receive the FULL input because ring_attn gather
        # already happened inside the encoder — every SP rank holds the full output.
        pg_mesh = stage_manager.pg_mesh
        my_modal = pg_mesh.my_modal

        if my_modal.model_name == self._ENC.model_name:
            input_list = [x_full.detach().clone()]
        else:
            input_list = [
                torch.zeros(num_microbatches * microbatch_size, _H, device="cuda")
            ]

        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "stage_manager": stage_manager,
        }
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=0))
        pp_ret = schedule.forward_backward_step(
            pp_model,
            iter(input_list),
            criterion,
            pp_optimizer,
            return_loss=True,
        )

        dist.barrier()

        # Loss check on the globally last stage.
        if stage_manager.is_last_stage(check_only_in_modal=False):
            torch.testing.assert_close(loss_ref, pp_ret["loss"], atol=1e-4, rtol=1e-4)

        if my_modal.model_name == self._ENC.model_name:
            # Both encoder SP ranks produced identical full-sequence outputs and
            # each received the same full gradient → individual grad equals baseline.
            torch.testing.assert_close(
                model_ref.enc.weight.grad,
                pp_model.enc.weight.grad,
                atol=1e-4,
                rtol=1e-4,
            )
        else:
            if stage_manager.stage_in_modal == 0:
                torch.testing.assert_close(
                    model_ref.llm0.weight.grad,
                    pp_model.llm0.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )
            else:
                torch.testing.assert_close(
                    model_ref.llm1.weight.grad,
                    pp_model.llm1.weight.grad,
                    atol=1e-4,
                    rtol=1e-4,
                )
