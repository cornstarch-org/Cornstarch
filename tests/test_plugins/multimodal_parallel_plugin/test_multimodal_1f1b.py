import os
import random
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import CudaAccelerator, get_accelerator
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalPipelineP2PCommunication,
    MultiModalPipelineStageManager,
    MultiModalProcessGroupMesh,
)

encoder1_template = PipelineTemplate("encoder1", [["layer.0", "layer.1", "layer.2"]])
encoder2_template = PipelineTemplate("encoder2", [["layer.0"], ["layer.1"]])
llm_template = PipelineTemplate(
    "llm", [["layer.0", "layer.1"], ["layer.2", "layer.3"], ["layer.4", "layer.5"]]
)


class P2PCommunicationClassBase(MultiProcessTestCase):
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


class TestHomogeneousTensorParallelMultiEncoderClass(P2PCommunicationClassBase):
    """
    Tests in this class has two parallel encoders and one LLM.
    Ranks at the border of
    """

    @property
    def world_size(self):
        return 24

    def create_p2p(
        self, tp_size: int
    ) -> tuple[MultiModalPipelineStageManager, MultimodalPipelineP2PCommunication]:
        encoder_templates = {
            encoder1_template: tp_size,
            encoder2_template: tp_size,
        }
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=(llm_template, tp_size),
        )
        stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
        p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

        return stage_manager, p2p

    @parametrize("tp_size", [1, 2, 4])
    def test_p2p_communication_forward_first(self, tp_size: int):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(backward_obj, is_broadcast=True)

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(backward_obj, is_broadcast=True)

                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

    @parametrize("tp_size", [1, 2, 4])
    def test_p2p_communication_backward_first(self, tp_size: int):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)

                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

    @parametrize("tp_size", [1, 2, 4])
    @parametrize("send_first", [True, False])
    def test_p2p_communication_coaleasced_forward_first(
        self, tp_size: int, send_first: bool
    ):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(backward_obj, is_broadcast=True)

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )
                recv_backward_objs = p2p.send_backward_recv_backward(
                    backward_obj, send_first=send_first, is_broadcast=True
                )

                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

    @parametrize("tp_size", [1, 2, 4])
    @parametrize("send_first", [True, False])
    def test_p2p_communication_coaleasced_backward_first(
        self, tp_size: int, send_first: bool
    ):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_backward_objs = p2p.send_backward_recv_backward(
                    backward_obj, send_first=send_first, is_broadcast=True
                )
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )

                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)


class TestHomogeneousTensorParallelSingleEncoderClass(P2PCommunicationClassBase):
    @property
    def world_size(self):
        return 16

    def create_p2p(
        self, tp_size: int
    ) -> tuple[MultiModalPipelineStageManager, MultimodalPipelineP2PCommunication]:
        encoder_templates = {encoder1_template: tp_size}

        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=(llm_template, tp_size),
        )
        stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
        p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

        return stage_manager, p2p

    @parametrize("tp_size", [1, 2, 4])
    def test_p2p_communication_forward_first(self, tp_size: int):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(backward_obj, is_broadcast=True)
                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(backward_obj, is_broadcast=True)
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj

    @parametrize("tp_size", [1, 2, 4])
    def test_p2p_communication_backward_first(self, tp_size: int):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()
                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj

    @parametrize("tp_size", [1, 2, 4])
    @parametrize("send_first", [True, False])
    def test_p2p_communication_coaleasced_forward_first(
        self, tp_size: int, send_first: bool
    ):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(backward_obj, is_broadcast=True)

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )
                recv_backward_objs = p2p.send_backward_recv_backward(
                    backward_obj, send_first=send_first, is_broadcast=True
                )

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj

    @parametrize("tp_size", [1, 2, 4])
    @parametrize("send_first", [True, False])
    def test_p2p_communication_coaleasced_backward_first(
        self, tp_size: int, send_first: bool
    ):
        stage_manager, p2p = self.create_p2p(tp_size)
        data = self.create_data()

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(backward_obj, is_broadcast=True)
                recv_forward_objs = p2p.recv_forward()

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)

                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj
            else:
                recv_backward_objs = p2p.send_backward_recv_backward(
                    backward_obj, send_first=send_first, is_broadcast=True
                )
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )

                assert len(recv_forward_objs) == 1
                assert recv_forward_objs[0] == forward_obj
                assert len(recv_backward_objs) == 1
                assert recv_backward_objs[0] == backward_obj


instantiate_parametrized_tests(TestHomogeneousTensorParallelSingleEncoderClass)
instantiate_parametrized_tests(TestHomogeneousTensorParallelMultiEncoderClass)
