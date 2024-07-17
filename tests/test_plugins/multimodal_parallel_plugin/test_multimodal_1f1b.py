import os
import random
import sys
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
encoder2_template = PipelineTemplate(
    "encoder2", [["layer.0"], ["layer.1"], ["layer.2"]]
)
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


class TestHomogeneousTensorParallelCommunicationClass(P2PCommunicationClassBase):
    @property
    def world_size(self):
        return 16

    @parametrize("tp_size", [1, 2, 4])
    def test_p2p_communication(self, tp_size: int):
        templates = {
            encoder1_template: tp_size,
            llm_template: tp_size,
        }
        execution_order = [(encoder1_template, llm_template)]

        pg_mesh = MultiModalProcessGroupMesh(
            modal_templates=templates,
            execution_order=execution_order,
        )
        stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
        p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

        rank = self.rank

        tensor = torch.ones(1, device=get_accelerator().get_current_device())
        data = [
            "tensor",
            tensor,
            [tensor],
            {"tensor": tensor},
        ]

        if stage_manager.is_first_stage(check_only_in_modal=False):
            for obj in data:
                p2p.send_forward(obj)

        elif stage_manager.is_last_stage(check_only_in_modal=False):
            for obj in data:
                recv_objs = p2p.recv_forward()
                assert len(recv_objs) == 1
                assert recv_objs[0] == obj

        else:
            for obj in data:
                recv_objs = p2p.recv_forward()
                p2p.send_forward(obj)


instantiate_parametrized_tests(TestHomogeneousTensorParallelCommunicationClass)
