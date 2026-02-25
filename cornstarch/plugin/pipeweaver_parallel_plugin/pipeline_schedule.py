from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

import torch
from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import PipelineP2PCommunication
from colossalai.pipeline.schedule._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
    merge_batch,
    model_forward,
    retain_grad,
    to_device,
    tree_map_hf,
)
from colossalai.pipeline.schedule.one_f_one_b import PipelineSchedule
from torch import nn
from torch.utils._pytree import tree_map

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
    MultimodalPipelineP2PCommunication,
)
from cornstarch.plugin.pipeweaver_parallel_plugin.pipeweaver_stage_manager import (
    PipeweaverPipelineStageManager,
)


class PipeweaverP2PCommunication(MultimodalPipelineP2PCommunication):
    """P2P communication for PipeWeaver.

    Reuses the multi-rank send/recv machinery of
    MultimodalPipelineP2PCommunication but accepts a
    PipeweaverPipelineStageManager (which is not a subclass of
    MultiModalPipelineStageManager).
    """

    def __init__(self, stage_manager: PipeweaverPipelineStageManager) -> None:
        assert isinstance(stage_manager, PipeweaverPipelineStageManager)
        PipelineP2PCommunication.__init__(self, stage_manager, overlap_p2p=False)


class PipeweaverEncoderTrainingPipeweaverScheduler(PipelineSchedule):
    """3-phase pipeline schedule for PipeWeaver co-located encoder + LLM.

    Phase 1 — All encoder forwards  (pipelined, M microbatches)
    Phase 2 — LLM 1F1B              (warmup → steady → cooldown)
    Phase 3 — All encoder backwards  (pipelined, M microbatches)

    Phase transitions rely on blocking P2P:
      * encoder last stage (rank N-1) sends M outputs to LLM first stage
        (rank 0) during Phase 1; rank 0 consumes them as Phase 2 warmup
        recv_forward calls.
      * LLM first stage (rank 0) sends M grads to encoder last stage
        (rank N-1) during Phase 2 cooldown; rank N-1 consumes them as
        Phase 3 recv_backward calls.
    """

    def __init__(
        self,
        stage_manager: PipeweaverPipelineStageManager,
        num_microbatches: int,
        microbatch_size: int,
    ) -> None:
        super().__init__(stage_manager)
        assert isinstance(stage_manager, PipeweaverPipelineStageManager)

        self.comm = PipeweaverP2PCommunication(stage_manager)
        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size

        self.batch: Optional[Any] = None
        self.batch_size: Optional[int] = None
        self.last_batch_size: Optional[int] = None
        self.microbatch_offset: Optional[int] = None

    # ------------------------------------------------------------------
    # Batch loading
    # ------------------------------------------------------------------

    def load_batch(self, data_iter: Iterable) -> None:
        batch = next(data_iter)
        self.microbatch_offset = 0
        self.batch = batch
        self.batch_size = get_batch_size(batch)

        if self.microbatch_size is None:
            assert self.batch_size % self.num_microbatches == 0
            self.microbatch_size = self.batch_size // self.num_microbatches
        if self.num_microbatches is None:
            assert self.batch_size % self.microbatch_size == 0
            self.num_microbatches = self.batch_size // self.microbatch_size

        if not self.forward_only:
            assert self.batch_size == self.microbatch_size * self.num_microbatches
            assert self.num_microbatches >= self.stage_manager.num_stages

        self.last_batch_size = self.batch_size

    def load_micro_batch(self) -> Any:
        assert self.microbatch_offset <= self.batch_size
        micro_batch = get_micro_batch(
            self.batch, self.microbatch_offset, self.microbatch_size
        )

        # Remove unnecessary padding
        num_tokens = max(torch.sum(micro_batch["attention_mask"], dim=1)).item()
        micro_batch["input_ids"] = micro_batch["input_ids"][:, :num_tokens]
        micro_batch["attention_mask"] = micro_batch["attention_mask"][:, :num_tokens]
        if "labels" in micro_batch:
            micro_batch["labels"] = micro_batch["labels"][:, :num_tokens]

        if "image_grid_thw" in micro_batch:
            previous_num_tokens = torch.sum(
                torch.prod(
                    self.batch["image_grid_thw"][: self.microbatch_offset], dim=1
                )
            ).item()
            current_num_tokens = torch.sum(
                torch.prod(micro_batch["image_grid_thw"], dim=1)
            ).item()
            micro_batch["pixel_values"] = self.batch["pixel_values"][
                previous_num_tokens : previous_num_tokens + current_num_tokens
            ]

        self.microbatch_offset += self.microbatch_size
        return tree_map(
            partial(to_device, device=get_accelerator().get_current_device()),
            micro_batch,
        )

    # ------------------------------------------------------------------
    # P2P helpers — thin wrappers that unpack single-element lists
    # ------------------------------------------------------------------

    def recv_forward(self) -> Any:
        if not self.stage_manager.get_prev_ranks():
            return None
        result = self.comm.recv_forward()
        assert isinstance(result, list) and len(result) == 1
        return result[0]

    def recv_backward(self) -> Any:
        if not self.stage_manager.get_next_ranks():
            return None
        result = self.comm.recv_backward()
        assert isinstance(result, list) and len(result) == 1
        return result[0]

    def send_forward(self, output_tensor: Any) -> None:
        if not self.stage_manager.get_next_ranks():
            return
        self.comm.send_forward(output_tensor, is_broadcast=True)

    def send_backward(self, input_tensor_grad: Any) -> None:
        if not self.stage_manager.get_prev_ranks():
            return
        self.comm.send_backward(input_tensor_grad, is_broadcast=True)

    def send_forward_recv_backward(
        self, output_tensor: Any, send_first: Optional[bool] = None
    ) -> Any:
        if not self.stage_manager.get_next_ranks():
            return None
        result = self.comm.send_forward_recv_backward(
            output_tensor, send_first=send_first, is_broadcast=True
        )
        return result[0]

    def send_backward_recv_forward(
        self, input_tensor_grad: Any, send_first: Optional[bool] = None
    ) -> Any:
        if not self.stage_manager.get_prev_ranks():
            return None
        result = self.comm.send_backward_recv_forward(
            input_tensor_grad, send_first=send_first, is_broadcast=True
        )
        assert isinstance(result, list) and len(result) == 1
        return result[0]

    def load_micro_batch(self) -> Any:
        """Load a micro batch from the current batch.
        Support Qwen2Vision.

        Returns:
            Any: Micro batch.
        """
        assert self.microbatch_offset <= self.batch_size, "Microbatches exhausted"
        micro_batch = get_micro_batch(
            self.batch, self.microbatch_offset, self.microbatch_size
        )

        if "image_grid_thw" in micro_batch:
            previous_num_tokens = torch.sum(
                torch.prod(
                    self.batch["image_grid_thw"][: self.microbatch_offset], dim=1
                )
            ).item()
            current_num_tokens = torch.sum(
                torch.prod(micro_batch["image_grid_thw"], dim=1)
            ).item()
            micro_batch["pixel_values"] = self.batch["pixel_values"][
                previous_num_tokens : previous_num_tokens + current_num_tokens
            ]
            self.microbatch_offset += self.microbatch_size
            return tree_map(
                partial(to_device, device=get_accelerator().get_current_device()),
                micro_batch,
            )

    # ------------------------------------------------------------------
    # Forward / backward primitives
    # ------------------------------------------------------------------

    def forward_step(
        self,
        model: nn.Module,
        input_obj: Optional[dict],
        criterion: Callable,
        accum_loss: Optional[torch.Tensor] = None,
        outputs: Optional[list[Any]] = None,
        compute_loss: bool = False,
    ) -> Union[torch.Tensor, dict]:
        micro_batch = self.load_micro_batch()
        if input_obj is not None and isinstance(micro_batch, dict):
            for key in input_obj.keys():
                micro_batch.pop(key, None)

        output_obj = model_forward(model, micro_batch, input_obj)

        if compute_loss and self.stage_manager.is_last_stage():
            loss = criterion(output_obj, micro_batch) / self.num_microbatches
            if accum_loss is not None:
                accum_loss.add_(loss.data)
            if outputs is not None:
                outputs.append(tree_map_hf(detach, output_obj))
            return loss
        return output_obj

    def backward_step(
        self,
        optimizer: OptimizerWrapper,
        input_obj: Optional[dict],
        output_obj: Union[dict, torch.Tensor],
        output_obj_grad: Optional[dict],
    ) -> Optional[dict]:
        tree_map(retain_grad, input_obj)

        if output_obj_grad is None:
            optimizer.backward(output_obj)
        else:
            keys = output_obj.get("backward_tensor_keys", output_obj_grad.keys())
            tensors_to_backward = []
            grads_to_backward = []
            for k in keys:
                if isinstance(output_obj[k], torch.Tensor):
                    tensors_to_backward.append(output_obj[k])
                    grads_to_backward.append(output_obj_grad[k])
            if len(tensors_to_backward) == 1:
                optimizer.backward_by_grad(tensors_to_backward[0], grads_to_backward[0])
            else:
                optimizer.backward_by_grad(tensors_to_backward, grads_to_backward)

        input_obj_grad = None
        if input_obj is not None:
            input_obj_grad = {}
            for k, v in input_obj.items():
                if isinstance(v, torch.Tensor) and v.grad is not None:
                    input_obj_grad[k] = v.grad
                elif isinstance(v, list):
                    input_obj_grad[k] = [item.grad for item in v]
        return input_obj_grad

    # ------------------------------------------------------------------
    # Main 3-phase schedule
    # ------------------------------------------------------------------

    def forward_backward_step(
        self,
        model: nn.Module,
        data_iter: Iterable,
        criterion: Callable[..., Any],
        optimizer: Optional[OptimizerWrapper] = None,
        return_loss: bool = False,
        return_outputs: bool = False,
    ) -> dict:
        self.forward_only = not torch.is_grad_enabled()
        if optimizer is None:
            assert self.forward_only

        self.load_batch(data_iter)

        stage = self.stage_manager.stage
        num_stages = self.stage_manager.num_stages
        M = self.num_microbatches

        accum_loss = None
        if return_loss and self.stage_manager.is_last_stage():
            accum_loss = torch.scalar_tensor(
                0, device=get_accelerator().get_current_device()
            )
        outputs = [] if return_outputs and self.stage_manager.is_last_stage() else None

        # ==============================================================
        # Phase 1 — All encoder forwards
        # ==============================================================
        self.stage_manager.set_encoder_mode()
        enc_input_objs: list[Any] = []
        enc_output_objs: list[Any] = []
        llm_prefetched_inputs: list[Any] = []

        for mb in range(M):
            input_obj = self.recv_forward()

            output_obj = self.forward_step(
                model, input_obj, criterion, compute_loss=False
            )
            if stage == 0:
                self.send_forward(output_obj)
                self.stage_manager.set_llm_mode()
                llm_input_obj = self.recv_forward()
                llm_prefetched_inputs.append(llm_input_obj)
                self.stage_manager.set_encoder_mode()
            else:
                self.send_forward(output_obj)
            enc_input_objs.append(input_obj)
            enc_output_objs.append(output_obj)

        # ==============================================================
        # Phase 2 — LLM 1F1B
        # ==============================================================
        self.stage_manager.set_llm_mode()
        self.microbatch_offset = 0

        num_warmup = min(num_stages - stage - 1, M)
        num_remaining = M - num_warmup

        llm_input_objs: list[Any] = []
        llm_output_objs: list[Any] = []
        border_input_obj_grads: list[Any] = []

        def recv_llm_input(mb: int) -> Any:
            if stage == 0:
                assert (
                    llm_prefetched_inputs
                ), "Missing prefetched border input on stage 0."
                return llm_prefetched_inputs.pop(0)
            return self.recv_forward()

        # --- warmup: pure forwards ---
        for i in range(num_warmup):
            input_obj = recv_llm_input(i)
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs, compute_loss=True
            )
            self.send_forward(output_obj)
            llm_input_objs.append(input_obj)
            llm_output_objs.append(output_obj)

        if num_remaining > 0:
            input_obj = recv_llm_input(num_warmup)

        # --- steady state: interleaved fwd + bwd ---
        for i in range(num_remaining):
            last_iteration = i == num_remaining - 1

            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs, compute_loss=True
            )
            output_obj_grad = self.send_forward_recv_backward(
                output_obj, send_first=(stage % 2 == 0)
            )

            llm_input_objs.append(input_obj)
            llm_output_objs.append(output_obj)

            input_obj = llm_input_objs.pop(0)
            output_obj = llm_output_objs.pop(0)
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )

            if stage == 0:
                border_input_obj_grads.append(input_obj_grad)
                if not last_iteration:
                    input_obj = recv_llm_input(num_warmup + i + 1)
            elif last_iteration:
                self.send_backward(input_obj_grad)
            else:
                input_obj = self.send_backward_recv_forward(
                    input_obj_grad, send_first=(stage % 2 == 0)
                )

        # --- cooldown: pure backwards ---
        for _ in range(num_warmup):
            input_obj = llm_input_objs.pop(0)
            output_obj = llm_output_objs.pop(0)
            output_obj_grad = self.recv_backward()
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )
            if stage == 0:
                border_input_obj_grads.append(input_obj_grad)
            else:
                self.send_backward(input_obj_grad)

        # ==============================================================
        # Phase 3 — All encoder backwards
        # ==============================================================
        self.stage_manager.set_encoder_mode()

        for mb in range(M):
            if stage == 0:
                self.stage_manager.set_llm_mode()
                self.send_backward(border_input_obj_grads[mb])
                self.stage_manager.set_encoder_mode()
                output_obj_grad = self.recv_backward()
            else:
                output_obj_grad = self.recv_backward()
            input_obj = enc_input_objs[mb]
            output_obj = enc_output_objs[mb]
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )
            self.send_backward(input_obj_grad)

        return {"loss": accum_loss, "outputs": outputs}
