from typing import Any, Optional

import torch
from colossalai.accelerator import get_accelerator
from colossalai.pipeline.p2p import (
    P2PMetadata,
    PipelineP2PCommunication,
    TensorMetadata,
    _create_recv_buffer,
    _cuda_safe_tensor_to_object,
    _filling_ops_queue,
    create_send_metadata,
)
from colossalai.pipeline.schedule.one_f_one_b import (
    OneForwardOneBackwardSchedule,
    PipelineSchedule,
)
from packaging.version import Version
from torch import distributed as dist
from torch.distributed import distributed_c10d as c10d
from torch.utils._pytree import tree_unflatten

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class MultimodalPipelineP2PCommunication(PipelineP2PCommunication):
    stage_manager: MultiModalPipelineStageManager

    def __init__(
        self, stage_manager: MultiModalPipelineStageManager, enable_metadata_cache: bool
    ):
        assert isinstance(stage_manager, MultiModalPipelineStageManager), (
            f"stage_manager must be an instance of MultiModalPipelineStageManager, "
            f"but got {type(stage_manager)}"
        )
        super().__init__(stage_manager, overlap_p2p=False)

    def _serialize_object(
        self,
        object_metadata: P2PMetadata,
        current_device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        send_object_tensor: torch.Tensor
        send_object_size_tensor: torch.Tensor
        if Version(torch.__version__) >= Version("1.13.0"):
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(
                object_metadata, device=current_device
            )
        else:
            send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(
                object_metadata
            )

        send_object_tensor = send_object_tensor.to(device=current_device)
        send_object_size_tensor = send_object_size_tensor.to(device=current_device)

        return send_object_tensor, send_object_size_tensor

    def _send_recv_metadata(
        self,
        object_metadata: Optional[P2PMetadata],
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
    ) -> list[P2PMetadata]:
        """
        Send and receive metadata from send_ranks and recv_ranks respectively.

        Optionally it can only send or receive metadata. To do it, set:
        - for send only:
            - recv_ranks = []
        - for receive only:
            - object_metadata = None
            - the value of send_ranks will be ignored

        Args:
            object_metadata (P2PMetadata): List of tensors to send.
            send_ranks (list[int]): List of ranks to send data to.
            recv_ranks (list[int]): List of ranks to receive data from.
            send_first (bool), optional: Whether to send data before receiving.

        Returns:
            list[list[torch.Tensor]]: List of received tensors.
                one list[torch.Tensor] per recv_rank in recv_ranks.
            []: an empty list if recv_tensor_metadata is empty (send only).
        """
        current_device = get_accelerator().get_current_device()
        ops: list[dist.Work] = []

        send_metadata_tensor: torch.Tensor = None
        send_metadata_size_tensor: torch.Tensor = None
        if object_metadata is not None:
            assert (
                send_ranks != []
            ), "send_ranks must be provided when object is not None"
            # NOTE: if object contains non-tensor objects, we have to send metadata
            send_metadata_tensor, send_metadata_size_tensor = self._serialize_object(
                object_metadata, current_device
            )
        else:
            # remove send_ranks as there is no data to send
            send_ranks = []

        recv_metadata_size_tensors = [
            torch.empty(1, dtype=torch.long, device=current_device) for _ in recv_ranks
        ]
        recv_metadata_tensors: list[torch.Tensor] = []

        # send and receive size first
        if send_first:
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_metadata_size_tensor, dist.isend, send_rank, ops, None
                )
            for recv_rank in recv_ranks:
                _filling_ops_queue(
                    recv_metadata_size_tensors, dist.irecv, recv_rank, ops, None
                )
        else:
            for recv_rank in recv_ranks:
                _filling_ops_queue(
                    recv_metadata_size_tensors, dist.irecv, recv_rank, ops, None
                )
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_metadata_size_tensor, dist.isend, send_rank, ops, None
                )

        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # create receive buffers based on the received size information
        recv_metadata_tensors = [
            torch.empty(
                recv_metadata_size_tensor.item(),
                dtype=torch.uint8,
                device=current_device,
            )
            for recv_metadata_size_tensor in recv_metadata_size_tensors
        ]

        ops.clear()
        # send and receive data
        if send_first:
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_metadata_tensor, dist.isend, send_rank, ops, None
                )
            for recv_rank, recv_metadata_tensor in zip(
                recv_ranks, recv_metadata_tensors
            ):
                _filling_ops_queue(
                    recv_metadata_tensor, dist.irecv, recv_rank, ops, None
                )
        else:
            for recv_rank, recv_metadata_tensor in zip(
                recv_ranks, recv_metadata_tensors
            ):
                _filling_ops_queue(
                    recv_metadata_tensor, dist.irecv, recv_rank, ops, None
                )
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_metadata_tensor, dist.isend, send_rank, ops, None
                )

        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # Unpick received metadata and return them
        unpickled_metadata: list[P2PMetadata] = []
        for recv_metadata_size_tensor, recv_metadata_tensor in zip(
            recv_metadata_size_tensors, recv_metadata_tensors
        ):
            recv_metadata_tensor = recv_metadata_tensor.to(
                dtype=torch.uint8, device=torch.device("cpu")
            )

            unpickle_object = _cuda_safe_tensor_to_object(
                recv_metadata_tensor, recv_metadata_size_tensor.item()
            )
            assert isinstance(unpickle_object, P2PMetadata)
            unpickled_metadata.append(unpickle_object)

        assert len(unpickled_metadata) == len(recv_ranks)
        return unpickled_metadata

    def _send_recv_tensors(
        self,
        send_tensor_objects: Optional[list[torch.Tensor]],
        recv_tensor_metadata: list[TensorMetadata],
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
    ) -> list[list[torch.Tensor]]:
        """
        Send and receive tensors from send_ranks and recv_ranks respectively.

        Optionally it can only send tensors or receive tensors. To do it, set:
        - for send only:
            - recv_tensor_metadata = []
            - the value of recv_ranks will be ignored
        - for receive only:
            - send_tensor_objects = None
            - the value of send_ranks will be ignored

        Args:
            send_tensor_objects (list[torch.Tensor]): List of tensors to send.
            recv_tensor_metadata (list[TensorMetadata]): List of metadata of tensors to receive.
            send_ranks (list[int]): List of ranks to send data to.
            recv_ranks (list[int]): List of ranks to receive data from.
            send_first (bool, optional): Whether to send data before receiving.

        Returns:
            list[list[torch.Tensor]]: List of received tensors.
                one list[torch.Tensor] per recv_rank in recv_ranks.
            []: an empty list if recv_tensor_metadata is empty (send only).
        """
        current_device = get_accelerator().get_current_device()

        if send_tensor_objects is None:
            # remove send_ranks as there is no data to send
            send_ranks = []

        recv_buffers: list[list[torch.Tensor]]
        if not recv_tensor_metadata:
            # remove recv_ranks as there is no data to receive
            recv_ranks = []
            recv_buffers = []
        else:
            assert len(recv_tensor_metadata) == len(recv_ranks)
            recv_buffers = [
                _create_recv_buffer(recv_metadata, current_device)
                for recv_metadata in recv_tensor_metadata
            ]

        ops: list[dist.Work] = []
        if send_first:
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_tensor_objects, dist.isend, send_rank, ops, None
                )
            for recv_rank, recv_buffer in zip(recv_ranks, recv_buffers):
                _filling_ops_queue(recv_buffer, dist.irecv, recv_rank, ops, None)
        else:
            for recv_rank, recv_buffer in zip(recv_ranks, recv_buffers):
                _filling_ops_queue(recv_buffer, dist.irecv, recv_rank, ops, None)
            for send_rank in send_ranks:
                _filling_ops_queue(
                    send_tensor_objects, dist.isend, send_rank, ops, None
                )

        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        return recv_buffers

    def _communicate(
        self,
        object: Any,
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
    ) -> Any:
        send_metadata: P2PMetadata = None
        send_tensor_objects: list[torch.Tensor] = None
        if object is not None:
            send_metadata, send_tensor_objects = create_send_metadata(
                object, strict=False, return_tensor=True
            )
        recv_metadata = self._send_recv_metadata(
            send_metadata, send_ranks, recv_ranks, send_first
        )
        recv_tensor_objects = self._send_recv_tensors(
            send_tensor_objects,
            [metadata.tensor_metadata for metadata in recv_metadata],
            send_ranks,
            recv_ranks,
            send_first,
        )

        received_objects: list[Any] = []
        for metadata, recv_tensor_object in zip(recv_metadata, recv_tensor_objects):
            assert isinstance(metadata, P2PMetadata)
            tree_spec = metadata.tree_spec
            non_tensor_object_indices = metadata.non_tensor_obj_idx
            non_tensor_objects = metadata.non_tensor_objs

            if recv_tensor_objects is None:
                recv_tensor_objects = []

            local_received_objects = []
            for idx in non_tensor_object_indices:
                local_received_objects.insert(idx, non_tensor_objects.pop(0))
            local_received_objects = tree_unflatten(recv_tensor_object, tree_spec)
            received_objects.append(local_received_objects)

        return received_objects

    def recv_forward(self) -> Any:
        input_tensors = self._communicate(
            object=None,
            send_ranks=[],
            recv_ranks=self.stage_manager.get_prev_ranks(),
        )

        return input_tensors

    def recv_backward(self) -> Any:
        output_tensor_grads = self._communicate(
            object=None,
            send_ranks=[],
            recv_ranks=self.stage_manager.get_next_ranks(),
        )

        return output_tensor_grads

    def send_forward(self, output_object: Any) -> None:
        self._communicate(
            object=output_object,
            send_ranks=self.stage_manager.get_next_ranks(),
            recv_ranks=[],
        )

    def send_backward(self, input_object: Any) -> None:
        self._communicate(
            object=input_object,
            send_ranks=self.stage_manager.get_prev_ranks(),
            recv_ranks=[],
        )

    def send_forward_recv_backward(self, output_object: Any, send_first: bool) -> Any:
        return self._communicate(
            object=output_object,
            send_ranks=self.stage_manager.get_next_ranks(),
            recv_ranks=self.stage_manager.get_prev_ranks(),
            send_first=send_first,
        )

    def send_backward_recv_forward(self, input_object: Any, send_first: bool) -> Any:
        return self._communicate(
            object=input_object,
            send_ranks=self.stage_manager.get_prev_ranks(),
            recv_ranks=self.stage_manager.get_next_ranks(),
            send_first=send_first,
        )


class MultimodalOneForwardOneBackwardSchedule(OneForwardOneBackwardSchedule):
    """
    1F1B pipeline schedule, with multi-modality in mind.

    In multimodal execution, a pipeline stage may have multiple senders and receivers.
    The Multimodal1F1BSchedule is designed to handle such cases.

    Args:
        stage_manager (MultiModalPipelineStageManager): "Multimodal" pipeline stage manager.
        num_microbatches(int): The number of microbatches.
        microbatch_size(int): Microbatch size.
    """

    def __init__(
        self,
        stage_manager: MultiModalPipelineStageManager,
        num_microbatches: int,
        microbatch_size: int,
        enable_metadata_cache: bool = True,
    ):
        assert (
            num_microbatches is not None and microbatch_size is not None
        ), "Both num_microbatches and microbatch_size must be provided."
        PipelineSchedule.__init__(self, stage_manager)

        self.comm: MultimodalPipelineP2PCommunication = (
            MultimodalPipelineP2PCommunication(stage_manager, enable_metadata_cache)
        )

        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

    def recv_forward(self, prev_rank: int = None) -> Any:
        if not isinstance(prev_rank, list):
            prev_rank = [prev_rank]

        return super().recv_forward(prev_rank)

    def recv_backward(self, next_rank: int = None) -> Any:
        return super().recv_backward(next_rank)

    def send_forward(self, output_tensor: Any, next_rank: int = None) -> None:
        return super().send_forward(output_tensor, next_rank)

    def send_backward(self, input_tensor_grad: Any, prev_rank: int = None) -> None:
        return super().send_backward(input_tensor_grad, prev_rank)

    def send_forward_recv_backward(
        self,
        output_tensor: Any,
        next_rank: int = None,
        send_prior_fallback: bool | None = None,
    ) -> Any:
        return super().send_forward_recv_backward(
            output_tensor, next_rank, send_prior_fallback
        )

    def send_backward_recv_forward(
        self,
        input_tensor_grad: Any,
        prev_rank: int = None,
        send_prior_fallback: bool | None = None,
    ) -> Any:
        return super().send_backward_recv_forward(
            input_tensor_grad, prev_rank, send_prior_fallback
        )
