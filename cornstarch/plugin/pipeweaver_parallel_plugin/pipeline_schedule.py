from __future__ import annotations

from functools import partial
from typing import Any, Callable, Iterable, Optional, Union

import torch
from colossalai.accelerator import get_accelerator
from colossalai.interface import OptimizerWrapper
from colossalai.pipeline.p2p import (
    PipelineP2PCommunication,
)
from colossalai.pipeline.schedule._utils import (
    detach,
    get_batch_size,
    get_micro_batch,
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

    def send_forward_recv_forward(
        self, output_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        result = super().send_forward_recv_forward(
            output_object, send_first=send_first, is_broadcast=is_broadcast
        )
        return result if result else None

    def send_backward_recv_backward(
        self, input_object: Any, send_first: bool, is_broadcast: bool
    ) -> Any:
        result = super().send_backward_recv_backward(
            input_object, send_first=send_first, is_broadcast=is_broadcast
        )
        return result if result else None


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
        #
        # Each rank runs its M encoder microbatches purely.  The last
        # encoder stage (rank N-1) sends its outputs to rank 0 via
        # encoder_next_ranks=[0], which is the same physical channel as
        # rank 0's llm_prev_ranks=[N-1].  Those messages accumulate in
        # the network buffer and are drained by recv_llm_input() in
        # Phase 2.  We must NOT recv them here on rank 0: doing so
        # creates a deadlock because rank N-1 cannot send until the
        # pipeline has filled (it needs mb data from ranks 1..N-2, which
        # in turn need rank 0 to keep sending encoder outputs — but rank
        # 0 would be blocked waiting for a recv that can never arrive).
        # ==============================================================
        self.stage_manager.set_encoder_mode()
        enc_input_objs: list[Any] = []
        enc_output_objs: list[Any] = []

        input_obj = self.recv_forward()

        for mb in range(M):
            output_obj = self.forward_step(
                model, input_obj, criterion, compute_loss=False
            )
            enc_input_objs.append(input_obj)
            enc_output_objs.append(output_obj)

            if self.stage_manager.is_last_stage():
                # Last encoder stage: buffer output (sent in bulk after Phase 1).
                if mb < M - 1:
                    input_obj = self.recv_forward()
            else:
                # Non-last encoder stage: pipeline send to next, recv from prev.
                # Rank 0 (encoder first stage) recv_forward returns None.
                self.send_forward(output_obj)
                if mb < M - 1:
                    input_obj = self.recv_forward()

        # Full tail flush: rank N-1 sends all M encoder outputs to rank 0 in bulk.
        # Rank 0 will pre-buffer these in Phase 2 before starting LLM work.
        # Still in encoder mode so get_next_ranks() = [0].
        if self.stage_manager.is_last_stage():
            for i in range(M):
                self.send_forward(enc_output_objs[i])

        # enc_grad_buffer is populated by rank N-1 during Phase 2 steady state
        # (live recvs from rank 0) and topped up in Phase 3 pre-buffer
        # (rank 0 cooldown grads bulk-flushed after Phase 2).
        enc_grad_buffer: list[Any] = []

        # ==============================================================
        # Phase 2 — LLM 1F1B
        # ==============================================================
        self.stage_manager.set_llm_mode()
        self.microbatch_offset = 0

        num_warmup = min(num_stages - stage - 1, M)
        num_remaining = M - num_warmup

        llm_input_objs: list[Any] = []
        llm_output_objs: list[Any] = []
        # Rank 0: cooldown grads buffered here and bulk-flushed to rank N-1 after Phase 2.
        border_input_obj_grads: list[Any] = []

        # Rank 0: pre-buffer all M LLM inputs from rank N-1 before any LLM work.
        # Rank N-1 is sending them in bulk (Phase 1→2 flush), so this drains that flush.
        llm_input_buffer: list[Any] = []
        if stage == 0:
            for _ in range(M):
                llm_input_buffer.append(self.recv_forward())

        llm_buf_idx = 0  # index into llm_input_buffer for rank 0

        def recv_llm_input() -> Any:
            nonlocal llm_buf_idx
            if stage == 0:
                obj = llm_input_buffer[llm_buf_idx]
                llm_buf_idx += 1
                return obj
            return self.recv_forward()

        # --- warmup: pure forwards ---
        for i in range(num_warmup):
            input_obj = recv_llm_input()
            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs, compute_loss=True
            )
            self.send_forward(output_obj)
            llm_input_objs.append(input_obj)
            llm_output_objs.append(output_obj)

        if num_remaining > 0:
            input_obj = recv_llm_input()

        # Rank N-1 starts live-receiving from rank 0 at steady iteration i_start.
        # i_start = num_stages // 2; total live recvs = M - (num_stages - 1).
        # Even num_stages: recv after send_forward_recv_backward, before backward_step.
        # Odd  num_stages: recv after send_backward_recv_forward, after backward_step.
        i_start = num_stages // 2
        live_recv_count = M - (num_stages - 1)

        # --- steady state: interleaved fwd + bwd ---
        for i in range(num_remaining):
            last_iteration = i == num_remaining - 1

            output_obj = self.forward_step(
                model, input_obj, criterion, accum_loss, outputs, compute_loss=True
            )
            output_obj_grad = self.send_forward_recv_backward(
                output_obj, send_first=(stage % 2 == 0)
            )

            # Even num_stages: rank N-1 live-recvs from rank 0 here (before bwd).
            if (
                stage == num_stages - 1
                and num_stages % 2 == 0
                and i_start <= i < i_start + live_recv_count
            ):
                self.stage_manager.set_encoder_mode()
                enc_grad_buffer.append(self.recv_backward())
                self.stage_manager.set_llm_mode()

            llm_input_objs.append(input_obj)
            llm_output_objs.append(output_obj)

            input_obj = llm_input_objs.pop(0)
            output_obj = llm_output_objs.pop(0)
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )

            if stage == 0:
                # Live send to rank N-1.  Rank N-1 is at its steady iteration
                # i+(num_stages//2), where it has already posted recv_backward.
                self.send_backward(input_obj_grad)
                if not last_iteration:
                    input_obj = recv_llm_input()
            elif stage == num_stages - 1:
                if last_iteration:
                    self.send_backward(input_obj_grad)
                else:
                    input_obj = self.send_backward_recv_forward(
                        input_obj_grad, send_first=(stage % 2 == 0)
                    )
                # Odd num_stages: rank N-1 live-recvs from rank 0 after send_bwd_recv_fwd.
                if num_stages % 2 == 1 and i_start <= i < i_start + live_recv_count:
                    self.stage_manager.set_encoder_mode()
                    enc_grad_buffer.append(self.recv_backward())
                    self.stage_manager.set_llm_mode()
            elif last_iteration:
                self.send_backward(input_obj_grad)
            else:
                input_obj = self.send_backward_recv_forward(
                    input_obj_grad, send_first=(stage % 2 == 0)
                )

        # --- cooldown: pure backwards ---
        for _i in range(num_warmup):
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
            if self.stage_manager.is_last_stage():
                output_obj_grad = enc_grad_buffer[mb]
            else:
                output_obj_grad = self.recv_backward()

            # Rank 0 sends cooldown grads; rank N-1 recvs them.
            # Both reach this point simultaneously: rank 0 enters Phase 3 after
            # num_stages-1 cooldown steps, while rank N-1 independently completes
            # num_stages-1 encoder backward steps (using buffered enc_grad_buffer).
            if mb == self.stage_manager.stage:
                if self.stage_manager.is_first_stage():
                    # LLM mode: send_backward uses llm_prev_ranks = [rank N-1]
                    self.stage_manager.set_llm_mode()
                    assert len(border_input_obj_grads) == num_stages - 1
                    for grad in border_input_obj_grads:
                        self.send_backward(grad)
                    self.stage_manager.set_encoder_mode()
                elif self.stage_manager.is_last_stage():
                    # Encoder mode: recv_backward uses encoder_next_ranks = [rank 0],
                    # which is the wrap-around channel matching rank 0's LLM send_backward.
                    for _ in range(num_stages - 1):
                        enc_grad_buffer.append(self.recv_backward())
                    assert len(enc_grad_buffer) == M

            input_obj = enc_input_objs[mb]
            output_obj = enc_output_objs[mb]
            input_obj_grad = self.backward_step(
                optimizer, input_obj, output_obj, output_obj_grad
            )
            self.send_backward(input_obj_grad)

        return {"loss": accum_loss, "outputs": outputs}
