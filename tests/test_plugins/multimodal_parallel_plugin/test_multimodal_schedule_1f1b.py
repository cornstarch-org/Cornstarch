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


class MultipleEncoderModelTestCaseBase:
    @property
    def world_size(self):
        return 6

    class DoubleEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
            self.encoder2 = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
            self.llm = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
            self.pp_config: dict = None

        def get_templates(self) -> list[PipelineTemplate]:
            return [
                PipelineTemplate(
                    "encoder1", [["encoder1.0", "encoder1.1", "encoder1.2"]]
                ),
                PipelineTemplate("encoder2", [["encoder2.0"], ["encoder2.1"]]),
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

                if modal_name == "encoder1":
                    assert start_idx >= 0 and end_idx <= 3
                    for layer in self.encoder1[start_idx:end_idx]:
                        x = layer(x)
                elif modal_name == "encoder2":
                    assert start_idx >= 0 and end_idx <= 2
                    for layer in self.encoder2[start_idx:end_idx]:
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
                x1 = copy.deepcopy(x)
                x2 = copy.deepcopy(x)
                for layer in self.encoder1:
                    x1 = layer(x1)
                for layer in self.encoder2:
                    x2 = layer(x2)
                x = torch.cat([x1, x2], dim=1)
                for layer in self.llm:
                    x = layer(x)

                return x

    def create_stage_manager(
        self, model: DoubleEncoderModel
    ) -> MultiModalPipelineStageManager:
        encoder1_template, encoder2_template, llm_template = model.get_templates()
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={encoder1_template: 1, encoder2_template: 1},
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

        model = self.SingleEncoderModel()
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

        input_list = [torch.rand(num_microbatches * microbatch_size, 8)]

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
class TestMultipleEncoderForwardBackwardOrderClass(
    MultipleEncoderModelTestCaseBase, ForwardBackwardOrderClassBase
):
    @property
    def world_size(self) -> int:
        return 6

    @parametrize("num_microbatches", [6, 12, 24], name_fn=lambda x: f"mb={x}")
    def test_forward_backward_order(self, num_microbatches: int):
        microbatch_size = 1
        call_list: list[str] = []

        model = self.DoubleEncoderModel().to("cuda")
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
        assert my_modal.model_name in ["encoder1", "encoder2", "llm"]
        if my_modal.model_name == "encoder1":
            self.register_hooks(model.encoder1[start_idx], call_list)
        elif my_modal.model_name == "encoder2":
            self.register_hooks(model.encoder2[start_idx], call_list)
        else:
            self.register_hooks(model.llm[start_idx], call_list)

        model.pp_config = {
            "modal_name": my_modal.model_name,
            "index": (start_idx, end_idx),
            "stage_manager": stage_manager,
        }

        input_list = [torch.rand(num_microbatches * microbatch_size, 4, 8).to("cuda")]

        schedule.forward_backward_step(
            model,
            iter(input_list),
            criterion,
            optimizer,
        )

        dist.barrier()
        torch.cuda.synchronize()

        # TODO: add assertions for the order of forward and backward calls
        assert len(call_list) == num_microbatches * 2

        # Encoder 1 has 1 stage, encoder 2 has 2 stages, llm has 3 stages
        # Maximum pipeline has 5 stages
        if self.rank == 0:
            expected_order = (
                ["forward"] * 4
                + ["backward", "forward"] * (num_microbatches - 4)
                + ["backward"] * 4
            )
        elif self.rank in [1, 2]:
            warmup = 5 - (self.rank - 1)
            expected_order = (
                ["forward"] * warmup
                + ["backward", "forward"] * (num_microbatches - warmup)
                + ["backward"] * warmup
            )
        else:
            warmup = 3 - (self.rank - 3)
            expected_order = (
                ["forward"] * warmup
                + ["backward", "forward"] * (num_microbatches - warmup)
                + ["backward"] * warmup
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
            assert torch.allclose(loss, pp_ret["loss"])

        # check gradients
        if my_modal.model_name == "encoder":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder[i].weight.grad,
                    pp_model.encoder[i].weight.grad,
                )
                assert torch.allclose(
                    model.encoder[i].bias.grad,
                    pp_model.encoder[i].bias.grad,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.llm[i].weight.grad,
                    pp_model.llm[i].weight.grad,
                )
                assert torch.allclose(
                    model.llm[i].bias.grad,
                    pp_model.llm[i].bias.grad,
                )

        # step
        model_optimizer.step()
        pp_optimizer.step()
        model_optimizer.zero_grad()
        pp_optimizer.zero_grad()

        # check updated param
        if my_modal.model_name == "encoder":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder[i].weight,
                    pp_model.encoder[i].weight,
                )
                assert torch.allclose(
                    model.encoder[i].bias,
                    pp_model.encoder[i].bias,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.llm[i].weight,
                    pp_model.llm[i].weight,
                )
                assert torch.allclose(
                    model.llm[i].bias,
                    pp_model.llm[i].bias,
                )


@instantiate_parametrized_tests
class TestScheduleMultipleEncoderClass(
    MultipleEncoderModelTestCaseBase, GlooDistributedTestBase
):
    @property
    def world_size(self) -> int:
        return 6

    @parametrize("num_microbatches", [6, 12, 24], name_fn=lambda x: f"mb={x}")
    @parametrize("microbatch_size", [1, 2, 4], name_fn=lambda x: f"mbs={x}")
    def test_schedule(self, num_microbatches: int, microbatch_size: int):
        model = self.DoubleEncoderModel().to("cuda")
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

        assert my_modal.model_name in ["encoder1", "encoder2", "llm"]
        pp_model.pp_config = {
            "modal_name": my_modal.model_name,
            "index": (start_idx, end_idx),
            "stage_manager": stage_manager,
        }

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1)
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=1))

        input_list = [
            torch.rand(num_microbatches * microbatch_size, 4, 8).to("cuda").to("cuda")
        ]
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
            assert torch.allclose(loss, pp_ret["loss"])

        # check gradients
        if my_modal.model_name == "encoder1":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder1[i].weight.grad,
                    pp_model.encoder1[i].weight.grad,
                )
                assert torch.allclose(
                    model.encoder1[i].bias.grad,
                    pp_model.encoder1[i].bias.grad,
                )
        elif my_modal.model_name == "encoder2":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder2[i].weight.grad,
                    pp_model.encoder2[i].weight.grad,
                )
                assert torch.allclose(
                    model.encoder2[i].bias.grad,
                    pp_model.encoder2[i].bias.grad,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.llm[i].weight.grad,
                    pp_model.llm[i].weight.grad,
                )
                assert torch.allclose(
                    model.llm[i].bias.grad,
                    pp_model.llm[i].bias.grad,
                )

        # step
        model_optimizer.step()
        pp_optimizer.step()
        model_optimizer.zero_grad()
        pp_optimizer.zero_grad()

        # check updated param
        if my_modal.model_name == "encoder1":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder1[i].weight,
                    pp_model.encoder1[i].weight,
                )
                assert torch.allclose(
                    model.encoder1[i].bias,
                    pp_model.encoder1[i].bias,
                )
        elif my_modal.model_name == "encoder2":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder2[i].weight,
                    pp_model.encoder2[i].weight,
                )
                assert torch.allclose(
                    model.encoder2[i].bias,
                    pp_model.encoder2[i].bias,
                )
        elif my_modal.model_name == "llm":
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.llm[i].weight,
                    pp_model.llm[i].weight,
                )
                assert torch.allclose(
                    model.llm[i].bias,
                    pp_model.llm[i].bias,
                )
