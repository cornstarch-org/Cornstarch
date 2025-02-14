import os
import re
from typing import Any
from unittest.mock import patch

import torch
from colossalai.accelerator import get_accelerator
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalPipelineP2PCommunication,
    MultiModalPipelineStageManager,
    MultiModalProcessGroupMesh,
)

from ...distributed_base import GlooDistributedTestBase
from ..common import encoder1_template, encoder2_template, llm_template_2stages


def create_data() -> list[Any]:
    tensor = torch.ones(1, device=get_accelerator().get_current_device())
    return [
        "tensor",
        tensor,
        [tensor],
        {"tensor": tensor},
    ]


def make_backward_object(
    backward_obj: Any,
    is_broadcast: bool,
    stage_manager: MultiModalPipelineStageManager,
):
    if is_broadcast:
        return backward_obj
    else:
        return [backward_obj for _ in range(len(stage_manager.get_prev_ranks()))]


@instantiate_parametrized_tests
class HomogeneousTensorParallelTestCase(GlooDistributedTestBase):
    @property
    def world_size(self):
        dp_size = 2
        tp_size = int(os.environ["TP_SIZE"])
        llm_sp_size = int(os.environ["LLM_SP_SIZE"])
        llm_pp_size = llm_template_2stages.num_stages
        num_encoders = int(os.environ["NUM_ENCODERS"])
        assert num_encoders in [1, 2]
        encoder_pp_size = (
            encoder1_template.num_stages
            if num_encoders == 1
            else encoder1_template.num_stages + encoder2_template.num_stages
        )
        return dp_size * tp_size * (encoder_pp_size + llm_sp_size * llm_pp_size)

    def setUp(self) -> None:
        # Extract tp_size, llm_sp_size, and num_encoders from the test method name
        pattern = r"tp=(\d+)_sp=(\d+)_enc=(\d+)"
        match = re.search(pattern, self._testMethodName)
        assert match is not None

        with patch.dict(
            os.environ,
            {
                "TP_SIZE": match.group(1),
                "LLM_SP_SIZE": match.group(2),
                "NUM_ENCODERS": match.group(3),
            },
        ):
            super().setUp()

    def create_p2p(
        self,
        tp_size: int,
        llm_sp_size: int = 1,
        num_encoders: int = 1,
    ) -> tuple[MultiModalPipelineStageManager, MultimodalPipelineP2PCommunication]:
        assert num_encoders in [1, 2]
        encoder_templates = (
            {
                encoder1_template: tp_size,
                encoder2_template: tp_size,
            }
            if num_encoders == 2
            else {
                encoder1_template: tp_size,
            }
        )
        pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates=encoder_templates,
            llm_template=(llm_template_2stages, tp_size, llm_sp_size),
        )

        stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
        p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

        return stage_manager, p2p

    @parametrize("tp_size", [1, 2, 4], name_fn=lambda x: f"tp={x}")
    @parametrize("llm_sp_size", [1, 2, 4], name_fn=lambda x: f"sp={x}")
    @parametrize("num_encoders", [1, 2], name_fn=lambda x: f"enc={x}")
    @parametrize("is_broadcast", [True, False], name_fn=lambda x: "bd" if x else "nbd")
    def test_p2p_communication_forward_first(
        self, tp_size: int, llm_sp_size: int, num_encoders: int, is_broadcast: bool
    ):
        """
        Test p2p communication without coalescing
        (using send_forward, recv_forward, send_backward, recv_backward)
        Send forward first.
        """
        stage_manager, p2p = self.create_p2p(tp_size, llm_sp_size, num_encoders)
        data = create_data()

        recv_forward_objs = None
        recv_backward_objs = None

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(
                    make_backward_object(backward_obj, is_broadcast, stage_manager),
                    is_broadcast=is_broadcast,
                )
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
            else:
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(
                    make_backward_object(backward_obj, is_broadcast, stage_manager),
                    is_broadcast=is_broadcast,
                )

            if recv_forward_objs:
                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

            if recv_backward_objs:
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)

    @parametrize("tp_size", [1, 2, 4], name_fn=lambda x: f"tp={x}")
    @parametrize("llm_sp_size", [1, 2, 4], name_fn=lambda x: f"sp={x}")
    @parametrize("num_encoders", [1, 2], name_fn=lambda x: f"enc={x}")
    @parametrize("is_broadcast", [True, False], name_fn=lambda x: "bd" if x else "nbd")
    def test_p2p_communication_backward_first(
        self, tp_size: int, llm_sp_size: int, num_encoders: int, is_broadcast: bool
    ):
        """
        Test p2p communication without coalescing
        (using send_forward, recv_forward, send_backward, recv_backward)
        Send backward first.
        """
        stage_manager, p2p = self.create_p2p(tp_size, llm_sp_size, num_encoders)
        data = create_data()

        recv_forward_objs = None
        recv_backward_objs = None

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(
                    make_backward_object(backward_obj, is_broadcast, stage_manager),
                    is_broadcast=is_broadcast,
                )
                recv_forward_objs = p2p.recv_forward()
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)
            else:
                recv_backward_objs = p2p.recv_backward()
                p2p.send_backward(
                    make_backward_object(backward_obj, is_broadcast, stage_manager),
                    is_broadcast=is_broadcast,
                )
                recv_forward_objs = p2p.recv_forward()
                p2p.send_forward(forward_obj, is_broadcast=True)

            if recv_forward_objs:
                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

            if recv_backward_objs:
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)

    @parametrize("tp_size", [1, 2, 4], name_fn=lambda x: f"tp={x}")
    @parametrize("llm_sp_size", [1, 2, 4], name_fn=lambda x: f"sp={x}")
    @parametrize("num_encoders", [1, 2], name_fn=lambda x: f"enc={x}")
    @parametrize("send_first", [True, False], name_fn=lambda x: "sf" if x else "rf")
    def test_p2p_communication_coalesced_forward_first(
        self, tp_size: int, llm_sp_size: int, num_encoders: int, send_first: bool
    ):
        """
        Test p2p communication with coalescing
        (using send_forward_recv_forward / send_backward_recv_backward)
        Send forward first.
        """
        stage_manager, p2p = self.create_p2p(tp_size, llm_sp_size, num_encoders)
        data = create_data()

        recv_forward_objs = None
        recv_backward_objs = None

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                recv_forward_objs = p2p.recv_forward()
                p2p.send_backward(backward_obj, is_broadcast=True)
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                p2p.send_forward(forward_obj, is_broadcast=True)
                recv_backward_objs = p2p.recv_backward()
            else:
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )
                recv_backward_objs = p2p.send_backward_recv_backward(
                    backward_obj, send_first=send_first, is_broadcast=True
                )

            if recv_forward_objs:
                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

            if recv_backward_objs:
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)

    @parametrize("tp_size", [1, 2, 4], name_fn=lambda x: f"tp={x}")
    @parametrize("llm_sp_size", [1, 2, 4], name_fn=lambda x: f"sp={x}")
    @parametrize("num_encoders", [1, 2], name_fn=lambda x: f"enc={x}")
    @parametrize("send_first", [True, False], name_fn=lambda x: "sf" if x else "rf")
    def test_p2p_communication_coalesced_backward_first(
        self, tp_size: int, llm_sp_size: int, num_encoders: int, send_first: bool
    ):
        """
        Test p2p communication with coalescing
        (using send_forward_recv_forward / send_backward_recv_backward)
        Send backward first.
        """
        stage_manager, p2p = self.create_p2p(tp_size, llm_sp_size, num_encoders)
        data = create_data()

        recv_forward_objs = None
        recv_backward_objs = None

        for forward_obj, backward_obj in zip(data, reversed(data)):
            if stage_manager.is_last_stage(check_only_in_modal=False):
                p2p.send_backward(
                    make_backward_object(backward_obj, True, stage_manager),
                    is_broadcast=True,
                )
                recv_forward_objs = p2p.recv_forward()
            elif stage_manager.is_first_stage(check_only_in_modal=False):
                recv_backward_objs = p2p.recv_backward()
                p2p.send_forward(forward_obj, is_broadcast=True)
            else:
                recv_backward_objs = p2p.send_backward_recv_backward(
                    make_backward_object(backward_obj, True, stage_manager),
                    send_first=send_first,
                    is_broadcast=True,
                )
                recv_forward_objs = p2p.send_forward_recv_forward(
                    forward_obj, send_first=send_first, is_broadcast=True
                )

            if recv_forward_objs:
                assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
                assert all(obj == forward_obj for obj in recv_forward_objs)

            if recv_backward_objs:
                assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
                assert all(obj == backward_obj for obj in recv_backward_objs)


# @instantiate_parametrized_tests
# class TestHomogeneousTensorParallelMultiEncoderClass(GlooDistributedTestBase):
#     """
#     Tests in this class has two parallel encoders and one LLM.
#     Ranks at the border of
#     """

#     @property
#     def world_size(self):
#         return 24

#     def create_p2p(
#         self, tp_size: int
#     ) -> tuple[MultiModalPipelineStageManager, MultimodalPipelineP2PCommunication]:
#         encoder_templates = {
#             encoder1_template: tp_size,
#             encoder2_template: tp_size,
#         }
#         pg_mesh = MultiModalProcessGroupMesh(
#             encoder_templates=encoder_templates,
#             llm_template=(llm_template, tp_size),
#         )
#         stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
#         p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

#         return stage_manager, p2p

#     def make_backward_object(
#         self,
#         backward_obj: Any,
#         is_broadcast: bool,
#         stage_manager: MultiModalPipelineStageManager,
#     ):
#         if is_broadcast:
#             return backward_obj
#         else:
#             return [backward_obj for _ in range(len(stage_manager.get_prev_ranks()))]

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("llm_sp_size", [1, 2, 4])
#     @parametrize("is_broadcast", [True, False])
#     def test_p2p_communication_forward_first(
#         self, tp_size: int, llm_sp_size: int, is_broadcast: bool
#     ):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     is_broadcast=is_broadcast,
#                 )

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     is_broadcast=is_broadcast,
#                 )

#                 assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
#                 assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
#                 assert all(obj == backward_obj for obj in recv_backward_objs)
#                 assert all(obj == forward_obj for obj in recv_forward_objs)

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("is_broadcast", [True, False])
#     def test_p2p_communication_backward_first(self, tp_size: int, is_broadcast: bool):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 p2p.send_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     is_broadcast=is_broadcast,
#                 )
#                 recv_forward_objs = p2p.recv_forward()

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     is_broadcast=is_broadcast,
#                 )
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)

#                 assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
#                 assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
#                 assert all(obj == backward_obj for obj in recv_backward_objs)
#                 assert all(obj == forward_obj for obj in recv_forward_objs)

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("send_first", [True, False])
#     @parametrize("is_broadcast", [True, False])
#     def test_p2p_communication_coaleasced_forward_first(
#         self, tp_size: int, send_first: bool, is_broadcast: bool
#     ):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_backward(backward_obj, is_broadcast=True)

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_forward_objs = p2p.send_forward_recv_forward(
#                     forward_obj, send_first=send_first, is_broadcast=True
#                 )
#                 recv_backward_objs = p2p.send_backward_recv_backward(
#                     backward_obj, send_first=send_first, is_broadcast=True
#                 )

#                 assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
#                 assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
#                 assert all(obj == backward_obj for obj in recv_backward_objs)
#                 assert all(obj == forward_obj for obj in recv_forward_objs)

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("send_first", [True, False])
#     @parametrize("is_broadcast", [True, False])
#     def test_p2p_communication_coaleasced_backward_first(
#         self, tp_size: int, send_first: bool, is_broadcast: bool
#     ):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 p2p.send_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     is_broadcast=is_broadcast,
#                 )
#                 recv_forward_objs = p2p.recv_forward()

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_backward_objs = p2p.send_backward_recv_backward(
#                     self.make_backward_object(
#                         backward_obj, is_broadcast, stage_manager
#                     ),
#                     send_first=send_first,
#                     is_broadcast=is_broadcast,
#                 )
#                 recv_forward_objs = p2p.send_forward_recv_forward(
#                     forward_obj, send_first=send_first, is_broadcast=True
#                 )

#                 assert len(stage_manager.get_prev_ranks()) == len(recv_forward_objs)
#                 assert len(stage_manager.get_next_ranks()) == len(recv_backward_objs)
#                 assert all(obj == backward_obj for obj in recv_backward_objs)
#                 assert all(obj == forward_obj for obj in recv_forward_objs)


# @instantiate_parametrized_tests
# class TestHomogeneousTensorParallelSingleEncoderClass(GlooDistributedTestBase):
#     @property
#     def world_size(self):
#         return 16

#     def create_p2p(
#         self, tp_size: int
#     ) -> tuple[MultiModalPipelineStageManager, MultimodalPipelineP2PCommunication]:
#         encoder_templates = {encoder1_template: tp_size}

#         pg_mesh = MultiModalProcessGroupMesh(
#             encoder_templates=encoder_templates,
#             llm_template=(llm_template, tp_size),
#         )
#         stage_manager = MultiModalPipelineStageManager(pg_mesh, pg_mesh.pp_axis)
#         p2p = MultimodalPipelineP2PCommunication(stage_manager=stage_manager)

#         return stage_manager, p2p

#     @parametrize("tp_size", [1, 2, 4])
#     def test_p2p_communication_forward_first(self, tp_size: int):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_backward(backward_obj, is_broadcast=True)
#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_backward(backward_obj, is_broadcast=True)
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj

#     @parametrize("tp_size", [1, 2, 4])
#     def test_p2p_communication_backward_first(self, tp_size: int):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 p2p.send_backward(backward_obj, is_broadcast=True)
#                 recv_forward_objs = p2p.recv_forward()
#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_backward(backward_obj, is_broadcast=True)
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("send_first", [True, False])
#     def test_p2p_communication_coaleasced_forward_first(
#         self, tp_size: int, send_first: bool
#     ):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 recv_forward_objs = p2p.recv_forward()
#                 p2p.send_backward(backward_obj, is_broadcast=True)

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 p2p.send_forward(forward_obj, is_broadcast=True)
#                 recv_backward_objs = p2p.recv_backward()

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_forward_objs = p2p.send_forward_recv_forward(
#                     forward_obj, send_first=send_first, is_broadcast=True
#                 )
#                 recv_backward_objs = p2p.send_backward_recv_backward(
#                     backward_obj, send_first=send_first, is_broadcast=True
#                 )

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj

#     @parametrize("tp_size", [1, 2, 4])
#     @parametrize("send_first", [True, False])
#     def test_p2p_communication_coaleasced_backward_first(
#         self, tp_size: int, send_first: bool
#     ):
#         stage_manager, p2p = self.create_p2p(tp_size)
#         data = create_data()

#         for forward_obj, backward_obj in zip(data, reversed(data)):
#             if stage_manager.is_last_stage(check_only_in_modal=False):
#                 p2p.send_backward(backward_obj, is_broadcast=True)
#                 recv_forward_objs = p2p.recv_forward()

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#             elif stage_manager.is_first_stage(check_only_in_modal=False):
#                 recv_backward_objs = p2p.recv_backward()
#                 p2p.send_forward(forward_obj, is_broadcast=True)

#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
#             else:
#                 recv_backward_objs = p2p.send_backward_recv_backward(
#                     backward_obj, send_first=send_first, is_broadcast=True
#                 )
#                 recv_forward_objs = p2p.send_forward_recv_forward(
#                     forward_obj, send_first=send_first, is_broadcast=True
#                 )

#                 assert len(recv_forward_objs) == 1
#                 assert recv_forward_objs[0] == forward_obj
#                 assert len(recv_backward_objs) == 1
#                 assert recv_backward_objs[0] == backward_obj
