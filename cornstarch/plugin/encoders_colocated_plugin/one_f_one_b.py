"""
Copied from colossalai.pipeline.schedule.one_f_one_b

add check_only_in_modal=False to is_first_stage and is_last_stage
Plus include the number of tokens from encoders to llm
"""

from typing import Any, Optional

import torch.cuda
from colossalai.pipeline.schedule.base import PipelineSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
    MultimodalPipelineP2PCommunication,
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
)


class MultimodalColocatedOneForwardOneBackwardSchedule(
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule
):

    def __init__(
        self,
        stage_manager: PipelineStageManager,
        num_microbatches: Optional[int] = None,
        microbatch_size: Optional[int] = None,
    ) -> None:
        """1F1B pipeline schedule.

        Args:
            stage_manager (PipelineStageManager): Pipeline stage manager
            num_microbatches (Optional[int], optional): The number of microbatches. If not provided, it will be derived from microbatch size. Defaults to None.
            microbatch_size (Optional[int], optional): Microbatch size. If num_microbatches is provided, this will be ignored. Defaults to None.
        """
        assert (
            num_microbatches is not None or microbatch_size is not None
        ), "Either num_microbatches or microbatch_size should be provided"
        PipelineSchedule.__init__(self, stage_manager)

        self.comm = MultimodalPipelineP2PCommunication(stage_manager)

        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

    def recv_forward(self) -> Any:
        input_tensors = None
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            input_tensors = self.comm.recv_forward()

            # all tensors are identical in colocated
            input_tensors = input_tensors[0]

        return input_tensors

    def recv_backward(self) -> Any:
        output_tensor_grads = None
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            output_tensor_grads = self.comm.recv_backward()
            output_tensor_grads = output_tensor_grads[0]
            # if not self.stage_manager.is_last_stage():
            #     # If receiver is in the same modal, unlist the output_tensor_grads
            #     assert (
            #         isinstance(output_tensor_grads, list)
            #         and len(output_tensor_grads) == 1
            #     )
            #     output_tensor_grads = output_tensor_grads[0]

        return output_tensor_grads

    def send_forward(self, output_tensor: Any) -> None:
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            self.comm.send_forward(output_tensor, is_broadcast=True)

    def send_backward(self, input_tensor: Any, input_tensor_grad: Any) -> None:
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            self.comm.send_backward(input_tensor_grad, is_broadcast=True)

    def send_forward_recv_backward(
        self, output_tensor: Any, send_first: Optional[bool] = None
    ) -> Any:
        """Sends the input tensor to the next stage and copy the gradient tensor from the next stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            next_rank (int, optional): The rank of the recipient of the tensor.
        """
        output_tensor_grads = None
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            output_tensor_grads = self.comm.send_forward_recv_backward(
                output_tensor, send_first=send_first, is_broadcast=True
            )
            output_tensor_grads = output_tensor_grads[0]

        return output_tensor_grads

    def send_backward_recv_forward(
        self,
        input_tensor: Any,
        input_tensor_grad: Any,
        send_first: Optional[bool] = None,
    ) -> Any:
        """Sends the gradient tensor to the previous stage and copy the input tensor from the previous stage in pipeline.
           For 1F1B.

        Args:
            output_object (Any): Object to be sent.
            prev_rank (int, optional): The rank of the recipient of the tensor.
        """
        input_tensors = None
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            input_tensors = self.comm.send_backward_recv_forward(
                input_tensor_grad,
                send_first=send_first,
                is_broadcast=True,
            )

            input_tensors = input_tensors[0]

        return input_tensors
