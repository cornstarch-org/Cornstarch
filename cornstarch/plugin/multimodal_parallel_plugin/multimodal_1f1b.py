from typing import Any, Callable, Iterable, Optional, Union

import torch
from colossalai.accelerator import get_accelerator
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.p2p import (
    P2PMetadata,
    PipelineP2PCommunication,
    TensorMetadata,
    _create_recv_buffer,
    _cuda_safe_tensor_to_object,
    _filling_ops_queue,
    create_send_metadata,
)
from colossalai.pipeline.schedule._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    retain_grad,
    to_device,
    tree_map,
    tree_map_hf,
)
from colossalai.pipeline.schedule.one_f_one_b import (
    OneForwardOneBackwardSchedule,
    PipelineSchedule,
)
from torch import distributed as dist
from torch import nn
from torch.distributed import distributed_c10d as c10d
from torch.utils._pytree import tree_unflatten

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class MultimodalPipelineP2PCommunication(PipelineP2PCommunication):
    stage_manager: MultiModalPipelineStageManager

    def __init__(self, stage_manager: MultiModalPipelineStageManager):
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
        send_object_tensor, send_object_size_tensor = c10d._object_to_tensor(
            object_metadata, device=current_device, group=dist.GroupMember.WORLD
        )

        send_object_tensor = send_object_tensor.to(device=current_device)
        send_object_size_tensor = send_object_size_tensor.to(device=current_device)

        return send_object_tensor, send_object_size_tensor

    def _send_recv_serialized_object(
        self,
        object_metadata: Optional[P2PMetadata | list[P2PMetadata]],
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
        is_broadcast: Optional[bool] = None,
    ) -> list[P2PMetadata]:
        """
        Send and receive metadata and potentially non-tensor objects from send_ranks and recv_ranks respectively.

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

        send_metadata_tensor: torch.Tensor | list[torch.Tensor] = []
        send_metadata_size_tensor: torch.Tensor | list[torch.Tensor] = []

        if object_metadata is not None:
            assert is_broadcast is not None
            assert (
                len(send_ranks) > 0
            ), "send_ranks must be provided when object is not None"

            if is_broadcast:
                send_metadata_tensor, send_metadata_size_tensor = (
                    self._serialize_object(object_metadata, current_device)
                )
            else:
                assert isinstance(object_metadata, list)
                assert len(object_metadata) == len(send_ranks)
                for metadata in object_metadata:
                    metadata_tensor, metadata_size_tensor = self._serialize_object(
                        metadata, current_device
                    )
                    send_metadata_tensor.append(metadata_tensor)
                    send_metadata_size_tensor.append(metadata_size_tensor)

        else:
            assert is_broadcast is None
            # remove send_ranks as there is no data to send
            send_ranks = []

        recv_metadata_size_tensors = [
            torch.empty(1, dtype=torch.long, device=current_device) for _ in recv_ranks
        ]
        recv_metadata_tensors: list[torch.Tensor] = []

        # send and receive size first
        if send_first:
            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_metadata_size_tensor, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, metadata_size_tenor in zip(
                    send_ranks, send_metadata_size_tensor
                ):
                    _filling_ops_queue(
                        metadata_size_tenor, dist.isend, send_rank, ops, None
                    )

            for recv_rank, recv_metadata_size_tensor in zip(
                recv_ranks, recv_metadata_size_tensors
            ):
                _filling_ops_queue(
                    recv_metadata_size_tensor, dist.irecv, recv_rank, ops, None
                )
        else:
            for recv_rank, recv_metadata_size_tensor in zip(
                recv_ranks, recv_metadata_size_tensors
            ):
                _filling_ops_queue(
                    recv_metadata_size_tensor, dist.irecv, recv_rank, ops, None
                )

            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_metadata_size_tensor, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, metadata_size_tenor in zip(
                    send_ranks, send_metadata_size_tensor
                ):
                    _filling_ops_queue(
                        metadata_size_tenor, dist.isend, send_rank, ops, None
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
            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_metadata_tensor, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, metadata_tensor in zip(send_ranks, send_metadata_tensor):
                    _filling_ops_queue(
                        metadata_tensor, dist.isend, send_rank, ops, None
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

            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_metadata_tensor, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, metadata_tensor in zip(send_ranks, send_metadata_tensor):
                    _filling_ops_queue(
                        metadata_tensor, dist.isend, send_rank, ops, None
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
        send_tensor_objects: Optional[list[torch.Tensor] | list[list[torch.Tensor]]],
        recv_tensor_metadata: list[TensorMetadata],
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
        is_broadcast: Optional[bool] = None,
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

        if send_tensor_objects is not None:
            assert is_broadcast is not None
            if not is_broadcast:
                assert isinstance(send_tensor_objects, list)
                assert len(send_tensor_objects) == len(send_ranks)
        else:
            assert is_broadcast is None
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
            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_tensor_objects, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, tensor_objects in zip(send_ranks, send_tensor_objects):
                    _filling_ops_queue(tensor_objects, dist.isend, send_rank, ops, None)

            for recv_rank, recv_buffer in zip(recv_ranks, recv_buffers):
                _filling_ops_queue(recv_buffer, dist.irecv, recv_rank, ops, None)
        else:
            for recv_rank, recv_buffer in zip(recv_ranks, recv_buffers):
                _filling_ops_queue(recv_buffer, dist.irecv, recv_rank, ops, None)

            if is_broadcast:
                for send_rank in send_ranks:
                    _filling_ops_queue(
                        send_tensor_objects, dist.isend, send_rank, ops, None
                    )
            elif is_broadcast is False:
                for send_rank, tensor_objects in zip(send_ranks, send_tensor_objects):
                    _filling_ops_queue(tensor_objects, dist.isend, send_rank, ops, None)

        if len(ops) > 0:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        return recv_buffers

    def _communicate(
        self,
        object: Optional[Any | list[Any]],
        send_ranks: list[int],
        recv_ranks: list[int],
        send_first: bool = True,
        is_broadcast: bool = None,
    ) -> Any:
        send_metadata: P2PMetadata | list[P2PMetadata] = None
        send_tensor_objects: list[torch.Tensor] | list[list[torch.Tensor]] = None

        if object is not None:
            assert is_broadcast is not None
            if is_broadcast:
                send_metadata, send_tensor_objects = create_send_metadata(
                    object, strict=False, return_tensor=True
                )
            else:
                send_metadata = []
                send_tensor_objects = []
                for obj in object:
                    metadata, tensor_objects = create_send_metadata(
                        obj, strict=False, return_tensor=True
                    )
                    send_metadata.append(metadata)
                    send_tensor_objects.append(tensor_objects)
        else:
            assert is_broadcast is None

        recv_metadata = self._send_recv_serialized_object(
            send_metadata, send_ranks, recv_ranks, send_first, is_broadcast
        )
        recv_tensor_objects = self._send_recv_tensors(
            send_tensor_objects,
            [metadata.tensor_metadata for metadata in recv_metadata],
            send_ranks,
            recv_ranks,
            send_first,
            is_broadcast,
        )

        received_objects: list[Any] = []
        for metadata, recv_tensor_object in zip(recv_metadata, recv_tensor_objects):
            assert isinstance(metadata, P2PMetadata)
            tree_spec = metadata.tree_spec
            non_tensor_object_indices = metadata.non_tensor_obj_idx
            non_tensor_objects = metadata.non_tensor_objs

            if recv_tensor_objects is None:
                recv_tensor_objects = []

            local_received_objects = recv_tensor_object
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

    def send_forward(self, output_object: Any, is_broadcast: bool) -> None:
        self._communicate(
            object=output_object,
            send_ranks=self.stage_manager.get_next_ranks(),
            recv_ranks=[],
            is_broadcast=is_broadcast,
        )

    def send_backward(self, input_object: Any, is_broadcast: bool) -> None:
        self._communicate(
            object=input_object,
            send_ranks=self.stage_manager.get_prev_ranks(),
            recv_ranks=[],
            is_broadcast=is_broadcast,
        )

    def send_forward_recv_backward(
        self, output_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        return self._communicate(
            object=output_object,
            send_ranks=self.stage_manager.get_next_ranks(),
            recv_ranks=self.stage_manager.get_next_ranks(),
            send_first=send_first,
            is_broadcast=is_broadcast,
        )

    def send_backward_recv_forward(
        self, input_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        return self._communicate(
            object=input_object,
            send_ranks=self.stage_manager.get_prev_ranks(),
            recv_ranks=self.stage_manager.get_prev_ranks(),
            send_first=send_first,
            is_broadcast=is_broadcast,
        )

    def send_forward_recv_forward(
        self, output_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        return self._communicate(
            object=output_object,
            send_ranks=self.stage_manager.get_next_ranks(),
            recv_ranks=self.stage_manager.get_prev_ranks(),
            send_first=send_first,
            is_broadcast=is_broadcast,
        )

    def send_backward_recv_backward(
        self, input_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        return self._communicate(
            object=input_object,
            send_ranks=self.stage_manager.get_prev_ranks(),
            recv_ranks=self.stage_manager.get_next_ranks(),
            send_first=send_first,
            is_broadcast=is_broadcast,
        )


class MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
    OneForwardOneBackwardSchedule
):
    """
    1F1B pipeline schedule, with multi-modality in mind.

    In multimodal execution, a pipeline stage may have multiple senders and receivers.
    The Multimodal1F1BSchedule is designed to handle such cases.

    Args:
        stage_manager (MultiModalPipelineStageManager): "Multimodal" pipeline stage manager.
        num_microbatches(int): The number of microbatches.
        microbatch_size(int): Microbatch size.
    """

    stage_manager: MultiModalPipelineStageManager

    def __init__(
        self,
        stage_manager: MultiModalPipelineStageManager,
        num_microbatches: int,
        microbatch_size: int,
    ):
        assert (
            num_microbatches is not None and microbatch_size is not None
        ), "Both num_microbatches and microbatch_size must be provided."
        PipelineSchedule.__init__(self, stage_manager)

        assert (
            not hasattr(stage_manager.pg_mesh, "decoder_templates")
            or len(stage_manager.pg_mesh.decoder_templates) == 0
        ), (
            "MultimodalEncoderTrainingOneForwardOneBackwardSchedule does not support "
            "decoders in the model."
        )

        self.comm: MultimodalPipelineP2PCommunication = (
            MultimodalPipelineP2PCommunication(stage_manager)
        )

        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size
        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

    def _merge_tensors(
        self, tensors_list: list[dict[str, torch.Tensor]], out_shapes: bool
    ) -> dict[str, list[torch.Tensor]]:
        assert isinstance(tensors_list, list)

        if len(tensors_list) == 0:
            return None

        assert all(isinstance(tensors, dict) for tensors in tensors_list)

        objects: dict[str, list[torch.Tensor]] = {}
        for tensor_dict in tensors_list:
            for key, tensor in tensor_dict.items():
                if key not in objects:
                    objects[key] = []
                objects[key].append(tensor)

        return objects

    def _split_tensors(
        self, tensors_dict: dict[str, list[torch.Tensor]]
    ) -> list[dict[str, torch.Tensor]]:
        assert len(tensors_dict) > 0

        objects: list[dict[str, torch.Tensor]] = [
            {} for _ in range(max(len(v) for v in tensors_dict.values()))
        ]

        for key, tensors in tensors_dict.items():
            for i, tensor in enumerate(tensors):
                objects[i][key] = tensor

        return objects

    def recv_forward(self) -> Any:
        input_tensors = None
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            input_tensors = self.comm.recv_forward()
            if not self.stage_manager.is_first_stage():
                # If sender is in the same modal, unlist the input_tensors
                assert isinstance(input_tensors, list) and len(input_tensors) == 1
                input_tensors = input_tensors[0]
            else:
                input_tensors = self._merge_tensors(input_tensors, out_shapes=True)

        return input_tensors

    def recv_backward(self) -> Any:
        output_tensor_grads = None
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            output_tensor_grads = self.comm.recv_backward()
            if not self.stage_manager.is_last_stage():
                # If receiver is in the same modal, unlist the output_tensor_grads
                assert (
                    isinstance(output_tensor_grads, list)
                    and len(output_tensor_grads) == 1
                )
                output_tensor_grads = output_tensor_grads[0]
            else:
                if isinstance(output_tensor_grads, list):
                    output_tensor_grads = output_tensor_grads[0]
                else:
                    output_tensor_grads = self._merge_tensors(
                        output_tensor_grads, out_shapes=False
                    )

        return output_tensor_grads

    def send_forward(self, output_tensor: Any) -> None:
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            self.comm.send_forward(output_tensor, is_broadcast=True)

    def send_backward(self, input_tensor: Any, input_tensor_grad: Any) -> None:
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            num_ranks_to_send = len(self.stage_manager.get_prev_ranks())
            if num_ranks_to_send > 1:
                self.comm.send_backward(
                    self._split_tensors(input_tensor_grad),
                    is_broadcast=False,
                )
            else:
                self.comm.send_backward(input_tensor_grad, is_broadcast=True)

    def send_forward_recv_backward(
        self, output_tensor: Any, send_first: Optional[bool] = None
    ) -> Any:
        output_tensor_grads = None
        if not self.stage_manager.is_last_stage(check_only_in_modal=False):
            output_tensor_grads = self.comm.send_forward_recv_backward(
                output_tensor, send_first=send_first, is_broadcast=True
            )
            if not self.stage_manager.is_last_stage():
                # If receiver is in the same modal, unlist the output_tensor_grads
                assert (
                    isinstance(output_tensor_grads, list)
                    and len(output_tensor_grads) == 1
                )
                output_tensor_grads = output_tensor_grads[0]
            else:
                # If there are multiple tensor grads and each has the same shape
                # with output_tensor, it means they are duplicated.
                if isinstance(output_tensor_grads, list):
                    output_tensor_grads = output_tensor_grads[0]
                else:
                    output_tensor_grads = self._merge_tensors(
                        output_tensor_grads, out_shapes=False
                    )

        return output_tensor_grads

    def send_backward_recv_forward(
        self,
        input_tensor: Any,
        input_tensor_grad: Any,
        send_first: Optional[bool] = None,
    ) -> Any:
        input_tensors = None
        if not self.stage_manager.is_first_stage(check_only_in_modal=False):
            num_ranks_to_send = len(self.stage_manager.get_prev_ranks())
            if num_ranks_to_send > 1:
                input_tensors = self.comm.send_backward_recv_forward(
                    self._split_tensors(input_tensor_grad),
                    send_first=send_first,
                    is_broadcast=False,
                )
            else:
                input_tensors = self.comm.send_backward_recv_forward(
                    input_tensor_grad,
                    send_first=send_first,
                    is_broadcast=True,
                )

            if not self.stage_manager.is_first_stage():
                # If sender is in the same modal, unlist the input_tensors
                assert isinstance(input_tensors, list) and len(input_tensors) == 1
                input_tensors = input_tensors[0]
            else:
                input_tensors = self._merge_tensors(input_tensors, out_shapes=True)

        return input_tensors

    def forward_step(
        self,
        model: nn.Module,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[list[Any]] = None,
    ) -> Union[torch.Tensor, dict]:
        """Forward one step of the pipeline

        Args:
            model (Module): Model to be run
            input_obj (Optional[dict]): The output from the previous stage. If it is the first stage, the `input_obj` is None.
            criterion (Callable): Criterion to calculate loss.
            accum_loss (Optional[torch.Tensor], optional): Accumulated loss. Defaults to None.
            outputs (Optional[List[Any]], optional): List to store the output of the last stage (final output). Defaults to None.

        Returns:
            Union[torch.Tensor, dict]: The intermediate output (dict) of the current stage. If it is the last stage, the output is the loss (Tensor).
        """
        micro_batch = self.load_micro_batch()
        # for the first stage, input_obj is None
        # for the non-first stage, input_obj is the output of the previous stage and it's must be a dict
        if input_obj is not None and isinstance(micro_batch, dict):
            for key in input_obj.keys():
                micro_batch.pop(key, None)

        output_obj = model_forward(model, micro_batch, input_obj)
        if self.stage_manager.is_last_stage(check_only_in_modal=False):
            loss = criterion(output_obj, micro_batch) / self.num_microbatches

            if accum_loss is not None:
                accum_loss.add_(loss.data)
            if outputs is not None:
                outputs.append(tree_map_hf(detach, output_obj))
            return loss
        else:
            return output_obj

    def backward_step(
        self,
        optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        """Backward one step of the pipeline

        Args:
            optimizer (OptimizerWrapper): Optimizer to update the model
            input_obj (Optional[dict]): Output of the previous stage. If it is the first stage, the `input_obj` is None.
            output_obj (Union[dict, torch.Tensor]): Output of the current stage. If it is the last stage, the output is the loss (Tensor).
            output_obj_grad (dict): Gradient of the `output_obj`. If it is the last stage, the `output_obj_grad` is None.

        Returns:
            Optional[dict]: Gradient of the `input_obj`. If it is the first stage, the `input_obj_grad` is None.
        """

        # Retain the grad on the input_obj.
        tree_map(retain_grad, input_obj)
        # Backward pass.
        if output_obj_grad is None:
            optimizer.backward(output_obj)
        else:
            keys = output_obj.get("backward_tensor_keys", output_obj_grad.keys())
            tensors_to_backward = []
            grads_to_backward = []
            for k in keys:
                tensors_to_backward.append(output_obj[k])
                grads_to_backward.append(output_obj_grad[k])
            if len(tensors_to_backward) == 1:
                optimizer.backward_by_grad(tensors_to_backward[0], grads_to_backward[0])
            else:
                optimizer.backward_by_grad(tensors_to_backward, grads_to_backward)

        # Collect the grad of the input_obj.
        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
                elif isinstance(v, list):
                    input_obj_grad[k] = [item.grad for item in v]

        return input_obj_grad

    def run_forward_backward(
        self,
        model: nn.Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: OptimizerWrapper | None = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        assert not self.forward_only

        self.load_batch(data_iter)

        my_modal = self.stage_manager.stage_index_to_modal[
            self.stage_manager.pg_mesh.coords[0][self.stage_manager.pipeline_axis]
        ]
        if isinstance(my_modal, list):
            my_modal = my_modal[0]

        # If LLM exists, calculate the number of warmup microbatches considering
        # LLM pipeline stages.
        if self.stage_manager.pg_mesh.llm_template is not None:
            llm_modal = self.stage_manager.pg_mesh.llm_template[0]

            if my_modal == llm_modal:
                num_warmup_microbatches = (
                    my_modal.num_stages - self.stage_manager.stage_in_modal - 1
                )
            else:
                num_warmup_microbatches = (
                    my_modal.num_stages
                    + llm_modal.num_stages
                    - self.stage_manager.stage_in_modal
                    - 1
                )

        else:
            num_warmup_microbatches = (
                my_modal.num_stages - self.stage_manager.stage_in_modal - 1
            )
        num_warmup_microbatches = min(num_warmup_microbatches, self.num_microbatches)
        num_microbatches_remaining = self.num_microbatches - num_warmup_microbatches

        # Input, output tensors only need to be saved when doing backward passes
        input_objs, output_objs = [], []

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage(check_only_in_modal=False):
            accum_loss = torch.scalar_tensor(
                0, device=get_accelerator().get_current_device()
            )
        outputs = (
            []
            if return_outputs
            and self.stage_manager.is_last_stage(check_only_in_modal=False)
            else None
        )

        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            input_obj = self.recv_forward()
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs
            )
            self.send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            input_obj = self.recv_forward()

        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            last_iteration = i == (num_microbatches_remaining - 1)

            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs
            )
            output_obj_grad = self.send_forward_recv_backward(
                output_obj, send_first=self.stage_manager.stage % 2 == 0
            )
            # Add input_obj and output_obj to end of list.
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            # Pop output_obj and output_obj from the start of the list for
            # the backward pass.
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )

            if last_iteration:
                self.send_backward(input_obj, input_obj_grad)
            else:
                input_obj = self.send_backward_recv_forward(
                    input_obj,
                    input_obj_grad,
                    send_first=self.stage_manager.stage % 2 == 0,
                )

        # Run cooldown backward passes.
        for i in range(num_warmup_microbatches):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)

            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )
            self.send_backward(input_obj, input_obj_grad)

        assert all(len(v) == 0 for v in input_objs) and all(
            len(v) == 0 for v in output_objs
        )

        if outputs is not None:
            if isinstance(model, ModelWrapper):
                model = model.unwrap()
            batch_size_dim = getattr(model, "batch_size_dim", 0)
            outputs = merge_batch(outputs, batch_size_dim)
        return {"loss": accum_loss, "outputs": outputs}
