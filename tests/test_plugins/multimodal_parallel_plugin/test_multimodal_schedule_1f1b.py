import copy
import functools
import os
import random
import sys
from types import MethodType
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import CudaAccelerator, get_accelerator
from colossalai.interface import OptimizerWrapper
from torch import nn
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
    MultiModalPipelineStageManager,
    MultiModalProcessGroupMesh,
)

# TODO (insujang): add tests for the cases without modal parallelism (modalities are colocated)


def pp_linear_fwd(
    forward,
    data: torch.Tensor = None,
    input_obj: torch.Tensor = None,
    stage_manager: MultiModalPipelineStageManager = None,
):
    if stage_manager.is_first_stage():
        return {"input_obj": forward(data)}
    elif stage_manager.is_last_stage():
        return forward(input_obj)
    else:
        return {"input_obj": forward(input_obj)}


class ScheduleTestClassBase(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    def create_data(self) -> list[Any]:
        tensor = torch.ones(1, device=get_accelerator().get_current_device())
        return [
            "tensor",
            tensor,
            [tensor],
            {"tensor": tensor},
        ]

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe) -> None:
        # Copy from: torch/testing/_internal/common_fsdp.py FSDPTest._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")
        backend = "gloo"

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=backend,
                world_size=self.world_size,
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier()

        with (
            torch.backends.cudnn.flags(
                enabled=True, deterministic=True, benchmark=True
            ),
            patch.object(
                CudaAccelerator, "get_current_device", return_value=torch.device("cpu")
            ),
        ):
            self.run_test(test_name, pipe)

        dist.barrier()

        dist.destroy_process_group()


class TestScheduleSingleEncoderClass(ScheduleTestClassBase):
    @property
    def world_size(self):
        return 4

    class SingleEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
            self.llm = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])

        def get_templates(self) -> list[PipelineTemplate]:
            return [
                PipelineTemplate("encoder", [["encoder.0", "encoder.1", "encoder.2"]]),
                PipelineTemplate(
                    "llm", [["llm.0", "llm.1"], ["llm.2", "llm.3"], ["llm.4", "llm.5"]]
                ),
            ]

        def forward(self, x):
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
            llm_template=(llm_template, 1),
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    @parametrize("num_microbatches", [4, 8, 12], name_fn=lambda x: f"mb={x}")
    @parametrize("microbatch_size", [1, 2, 4], name_fn=lambda x: f"mbs={x}")
    def test_schedule(self, num_microbatches: int, microbatch_size: int):
        model = self.SingleEncoderModel()
        pp_model = copy.deepcopy(model)

        stage_manager = self.create_stage_manager(model=model)
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager, num_microbatches, microbatch_size
        )

        start_idx, end_idx = stage_manager.get_stage_index(
            stage_manager.distribute_layers()
        )

        def pp_forward(
            self: TestScheduleSingleEncoderClass.SingleEncoderModel,
            x,
            start_idx: int,
            end_idx: int,
        ):
            if start_idx < 3:
                assert start_idx >= 0 and end_idx <= 3
                for layer in self.encoder[start_idx:end_idx]:
                    x = layer(x)
            else:
                assert start_idx >= 3 and end_idx <= 9
                for layer in self.llm[start_idx - 3 : end_idx - 3]:
                    x = layer(x)
            return x

        pp_model._forward = MethodType(
            functools.partial(pp_forward, start_idx=start_idx, end_idx=end_idx),
            pp_model,
        )
        pp_model.forward = MethodType(
            functools.partial(pp_linear_fwd, stage_manager=stage_manager),
            pp_model._forward,
        )

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1)
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=1))

        def criterion(x, *args, **kwargs):
            return (x * x).mean()

        input_list = [torch.rand(num_microbatches * microbatch_size, 8)]
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

        if stage_manager.is_last_stage():
            assert torch.allclose(loss, pp_ret["loss"])

        # check gradients
        if start_idx < 3:
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder[i].weight.grad,
                    pp_model.encoder[i].weight.grad,
                )
                assert torch.allclose(
                    model.encoder[i].bias.grad,
                    pp_model.encoder[i].bias.grad,
                )
        else:
            for i in range(start_idx - 3, end_idx - 3):
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
        if start_idx < 3:
            for i in range(start_idx, end_idx):
                assert torch.allclose(
                    model.encoder[i].weight,
                    pp_model.encoder[i].weight,
                )
                assert torch.allclose(
                    model.encoder[i].bias,
                    pp_model.encoder[i].bias,
                )
        else:
            for i in range(start_idx - 3, end_idx - 3):
                assert torch.allclose(
                    model.llm[i].weight,
                    pp_model.llm[i].weight,
                )
                assert torch.allclose(
                    model.llm[i].bias,
                    pp_model.llm[i].bias,
                )


class TestScheduleMultipleEncoderClass(ScheduleTestClassBase):
    @property
    def world_size(self):
        return 6

    class DoubleEncoderModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = nn.ModuleList([nn.Linear(8, 8) for _ in range(3)])
            self.encoder2 = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
            self.llm = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])

        def get_templates(self) -> list[PipelineTemplate]:
            return [
                PipelineTemplate(
                    "encoder1", [["encoder1.0", "encoder1.1", "encoder1.2"]]
                ),
                PipelineTemplate("encoder2", [["encoder2.0"], ["encoder2.1"]]),
                PipelineTemplate(
                    "llm", [["llm.0", "llm.1"], ["llm.2", "llm.3"], ["llm.4", "llm.5"]]
                ),
            ]

        def forward(self, x):
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
            llm_template=(llm_template, 1),
        )
        return MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)

    def test_schedule(self, num_microbatches: int = 6, microbatch_size: int = 1):
        model = self.DoubleEncoderModel()
        pp_model = copy.deepcopy(model)

        stage_manager = self.create_stage_manager(model=model)
        schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            stage_manager, num_microbatches, microbatch_size
        )

        start_idx, end_idx = stage_manager.get_stage_index(
            stage_manager.distribute_layers()
        )

        def pp_forward(
            self: TestScheduleMultipleEncoderClass.DoubleEncoderModel,
            x,
            start_idx: int,
            end_idx: int,
        ):
            if start_idx < 3:
                assert start_idx >= 0 and end_idx <= 3
                for layer in self.encoder1[start_idx:end_idx]:
                    x = layer(x)
            elif start_idx < 5:
                assert start_idx >= 3 and end_idx <= 5
                for layer in self.encoder2[start_idx - 3 : end_idx - 3]:
                    x = layer(x)
            else:
                assert start_idx >= 5 and end_idx <= 11
                for layer in self.llm[start_idx - 5 : end_idx - 5]:
                    x = layer(x)

            return x

        pp_model._forward = MethodType(
            functools.partial(pp_forward, start_idx=start_idx, end_idx=end_idx),
            pp_model,
        )
        pp_model.forward = MethodType(
            functools.partial(pp_linear_fwd, stage_manager=stage_manager),
            pp_model._forward,
        )

        model_optimizer = torch.optim.SGD(model.parameters(), lr=1)
        pp_optimizer = OptimizerWrapper(torch.optim.SGD(pp_model.parameters(), lr=1))

        def criterion(x, *args, **kwargs):
            return (x * x).mean()

        input_list = [torch.rand(num_microbatches * microbatch_size, 4, 8)]
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

        if stage_manager.is_last_stage():
            assert torch.allclose(loss, pp_ret["loss"])


instantiate_parametrized_tests(TestScheduleSingleEncoderClass)
instantiate_parametrized_tests(TestScheduleMultipleEncoderClass)
