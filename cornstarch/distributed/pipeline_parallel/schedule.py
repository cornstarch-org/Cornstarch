"""Training schedules for distributed execution.

Provides a unified ``step()`` interface for both pipeline-parallel (PP)
and non-pipeline-parallel training.  Both schedules accept a
``CornstarchExecutionPlan`` and an ``ExecutionFuture`` that describe
the multimodal computation DAG; the schedule determines *how* to
execute it (single-shot vs. microbatch-pipelined 1F1B).

``TrainingSchedule``
    Abstract base: execute one forward-backward training step.

``NonPipelineParallelSchedule``
    Executes the full DAG on every rank in a single forward pass,
    then runs backward.  Used when PP is disabled.

``OneForwardOneBackwardSchedule``
    Splits the batch into microbatches and pipelines them across PP
    stages using the 1F1B schedule.  Pre-pipeline nodes (modality
    encoders, merge) run on the first stage; the language model is
    pipelined across stages.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from cornstarch.distributed.pipeline_parallel.p2p import PipelineP2PCommunication
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.models.multimodal.execution import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    ExecutionNode,
)


# ---------------------------------------------------------------------------
# Tree utilities for nested tensor structures
# ---------------------------------------------------------------------------

def _tree_map(fn: Callable, obj: Any) -> Any:
    """Apply ``fn`` to every tensor in a nested dict / list / tuple."""
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_map(fn, v) for v in obj)
    return obj


def _detach(obj: Any) -> Any:
    """Detach all tensors from the computation graph."""
    return _tree_map(lambda t: t.detach(), obj)


def _first_tensor_device(obj: Any) -> torch.device | None:
    """Return the device of the first tensor found in a nested structure."""
    if isinstance(obj, torch.Tensor):
        return obj.device
    if isinstance(obj, dict):
        for value in obj.values():
            device = _first_tensor_device(value)
            if device is not None:
                return device
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            device = _first_tensor_device(value)
            if device is not None:
                return device
    return None


def _retain_grad(obj: Any) -> None:
    """Call ``retain_grad()`` on tensors that require gradients."""
    def _rg(t: torch.Tensor) -> torch.Tensor:
        if t.requires_grad:
            t.retain_grad()
        return t
    _tree_map(_rg, obj)


def _get_micro_batch(batch: Any, offset: int, size: int) -> Any:
    """Slice a micro-batch from ``batch`` along dimension 0."""
    if isinstance(batch, torch.Tensor):
        return batch[offset : offset + size]
    if isinstance(batch, dict):
        return {k: _get_micro_batch(v, offset, size) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_get_micro_batch(v, offset, size) for v in batch)
    return batch


def _merge_batch(batches: list[Any], dim: int = 0) -> Any:
    """Concatenate a list of batches along ``dim``."""
    if not batches:
        return None
    if isinstance(batches[0], torch.Tensor):
        return torch.cat(batches, dim=dim)
    if isinstance(batches[0], dict):
        return {k: _merge_batch([b[k] for b in batches], dim) for k in batches[0]}
    return batches


def _model_forward(
    model: nn.Module,
    micro_batch: dict,
    input_obj: Any | None,
) -> Any:
    """Merge ``input_obj`` into ``micro_batch`` and call the model."""
    kwargs = dict(micro_batch) if isinstance(micro_batch, dict) else {"input": micro_batch}
    if input_obj is not None:
        if isinstance(input_obj, dict):
            for k in input_obj:
                kwargs.pop(k, None)
            kwargs.update(input_obj)
        else:
            kwargs["hidden_states"] = input_obj
    return model(**kwargs)


# ---------------------------------------------------------------------------
# TrainingSchedule base
# ---------------------------------------------------------------------------

class TrainingSchedule(ABC):
    """Execute one forward-backward training step and return results.

    Subclasses implement ``step()`` which runs forward, computes loss,
    runs backward, and returns a dict with ``"loss"`` and ``"outputs"``.
    The caller is responsible for gradient synchronization, optimizer
    step, and zeroing gradients after ``step()`` returns.
    """

    @abstractmethod
    def step(
        self,
        batch: dict[str, torch.Tensor],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        """Execute one training step.

        Returns a dict with:
        - ``"loss"``: accumulated loss tensor or ``None``
        - ``"outputs"``: per-microbatch outputs or ``None``
        """


# ---------------------------------------------------------------------------
# Non-PP schedule
# ---------------------------------------------------------------------------

class NonPipelineParallelSchedule(TrainingSchedule):
    """Executes the full execution plan DAG on every rank, then backward.

    Used when pipeline parallelism is disabled.  The entire multimodal
    computation (encoder → merge → language model) runs in a single
    forward pass on each rank, followed by a single backward pass.
    """

    def __init__(
        self,
        plan: CornstarchExecutionPlan,
        output_future: ExecutionFuture,
    ) -> None:
        self._plan = plan
        self._output_future = output_future

    def step(
        self,
        batch: dict[str, torch.Tensor],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        output = self._output_future.execute(inputs=batch)
        loss = criterion(output, batch)
        loss.backward()
        return {
            "loss": loss.detach() if return_loss else None,
            "outputs": output if return_outputs else None,
        }


# ---------------------------------------------------------------------------
# PP stage model auto-built from the DAG
# ---------------------------------------------------------------------------

class _StageModel(nn.Module):
    """Stage-aware model auto-built from execution plan nodes.

    On the first PP stage, executes all pre-pipeline nodes (modality
    encoders, merge) using the execution plan's node-execution logic,
    then runs the language model.  On later stages, passes received
    hidden states directly to the language model.
    """

    def __init__(
        self,
        plan: CornstarchExecutionPlan,
        pre_pipeline_nodes: list[ExecutionNode],
        language_model_node: ExecutionNode,
        is_first_stage: bool,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._pre_pipeline_nodes = pre_pipeline_nodes
        self._lm_node = language_model_node
        self._is_first_stage = is_first_stage

    def forward(self, **kwargs) -> Any:
        if "hidden_states" in kwargs:
            # Non-first PP stage: received hidden_states from P2P.
            # Pass through to the LM which has a PipelineParallelForwardSpec
            # that extracts hidden_states in embed_inputs().
            lm = self._lm_node.params["module"]
            return lm(**kwargs)

        # First PP stage (or single-stage): execute pre-pipeline nodes
        # (encoders, merge) then the LM node.
        values: dict[str, Any] = dict(kwargs)
        for node in self._pre_pipeline_nodes:
            values[node.name] = CornstarchExecutionPlan._execute_node(node, values)
        return CornstarchExecutionPlan._execute_node(self._lm_node, values)


# ---------------------------------------------------------------------------
# 1F1B PP schedule
# ---------------------------------------------------------------------------

class OneForwardOneBackwardSchedule(TrainingSchedule):
    """Pipeline-parallel 1F1B schedule driven by a CornstarchExecutionPlan.

    Inspects the execution plan's DAG to identify pre-pipeline nodes
    (modality encoders, merge) and the pipeline node (language model),
    then auto-builds a stage-aware model wrapper.  The schedule splits
    the batch into microbatches and pipelines them across PP stages
    using the standard 1F1B pattern: warmup, steady state, cooldown.
    """

    def __init__(
        self,
        plan: CornstarchExecutionPlan,
        output_future: ExecutionFuture,
        mesh: ModalProcessGroupMesh,
        num_microbatches: int,
        microbatch_size: int,
    ) -> None:
        self._mesh = mesh
        self._comm = PipelineP2PCommunication(mesh)
        self._num_microbatches = num_microbatches
        self._microbatch_size = microbatch_size

        # Split DAG into pre-pipeline and pipeline nodes.
        nodes = plan._topological_nodes(output_future.name)
        lm_node = next(n for n in nodes if n.kind == "run_language_model")
        pre_nodes = [n for n in nodes if n.kind != "run_language_model"]

        self._stage_model = _StageModel(
            plan, pre_nodes, lm_node, mesh.is_first_stage()
        )

    def step(
        self,
        batch: dict[str, torch.Tensor],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        """Execute one 1F1B training step (forward + backward)."""
        self._batch = batch
        self._microbatch_offset = 0
        return self._run_1f1b(criterion, optimizer, return_loss, return_outputs)

    # ------------------------------------------------------------------
    # Microbatch management
    # ------------------------------------------------------------------

    def _load_micro_batch(self) -> Any:
        mb = _get_micro_batch(
            self._batch, self._microbatch_offset, self._microbatch_size
        )
        self._microbatch_offset += self._microbatch_size
        return mb

    # ------------------------------------------------------------------
    # Forward / backward steps
    # ------------------------------------------------------------------

    def _forward_step(
        self,
        input_obj: Any | None,
        criterion: Callable,
        accum_loss: torch.Tensor | None = None,
        outputs: list[Any] | None = None,
    ) -> Any:
        """Run one micro-batch forward and compute loss on the last stage."""
        micro_batch = self._load_micro_batch()

        if input_obj is not None and isinstance(micro_batch, dict):
            if isinstance(input_obj, dict):
                for k in input_obj:
                    micro_batch.pop(k, None)

        output_obj = _model_forward(self._stage_model, micro_batch, input_obj)

        if self._mesh.is_last_stage():
            loss = criterion(output_obj, micro_batch) / self._num_microbatches
            if accum_loss is not None:
                accum_loss.add_(loss.detach())
            if outputs is not None:
                outputs.append(_detach(output_obj))
            return loss

        return output_obj

    def _backward_step(
        self,
        optimizer: Optimizer | None,
        input_obj: Any | None,
        output_obj: Any,
        output_obj_grad: Any | None,
    ) -> Any | None:
        """Run one micro-batch backward and return input gradients."""
        _retain_grad(input_obj)

        if output_obj_grad is None:
            output_obj.backward()
        else:
            keys = None
            if isinstance(output_obj, dict):
                keys = output_obj.get("backward_tensor_keys") or list(
                    output_obj_grad.keys()
                )
            if keys is not None:
                tensors = [output_obj[k] for k in keys if isinstance(output_obj[k], torch.Tensor)]
                grads = [output_obj_grad[k] for k in keys if isinstance(output_obj[k], torch.Tensor)]
                if len(tensors) == 1:
                    torch.autograd.backward(tensors[0], grads[0])
                else:
                    torch.autograd.backward(tensors, grads)
            else:
                torch.autograd.backward(output_obj, output_obj_grad)

        input_obj_grad: dict | None = None
        if input_obj is not None:
            input_obj_grad = {}
            if isinstance(input_obj, dict):
                for k, v in input_obj.items():
                    if isinstance(v, torch.Tensor) and v.grad is not None:
                        input_obj_grad[k] = v.grad
            elif isinstance(input_obj, torch.Tensor) and input_obj.grad is not None:
                input_obj_grad = input_obj.grad

        return input_obj_grad

    # ------------------------------------------------------------------
    # P2P wrappers
    # ------------------------------------------------------------------

    def _recv_forward(self) -> Any | None:
        if self._mesh.is_first_stage():
            return None
        return self._comm.recv_forward()

    def _recv_backward(self) -> Any | None:
        if self._mesh.is_last_stage():
            return None
        return self._comm.recv_backward()

    def _send_forward(self, output: Any) -> None:
        if self._mesh.is_last_stage():
            return
        self._comm.send_forward(output)

    def _send_backward(self, grad: Any) -> None:
        if self._mesh.is_first_stage():
            return
        self._comm.send_backward(grad)

    def _send_forward_recv_backward(
        self, output: Any, send_first: bool = True
    ) -> Any | None:
        if self._mesh.is_last_stage():
            return None
        return self._comm.send_forward_recv_backward(output, send_first)

    def _send_backward_recv_forward(
        self, grad: Any, send_first: bool = True
    ) -> Any | None:
        if self._mesh.is_first_stage():
            return None
        return self._comm.send_backward_recv_forward(grad, send_first)

    # ------------------------------------------------------------------
    # Main 1F1B loop
    # ------------------------------------------------------------------

    def _run_1f1b(
        self,
        criterion: Callable[..., Any],
        optimizer: Optimizer | None,
        return_loss: bool,
        return_outputs: bool,
    ) -> dict:
        num_warmup = min(
            self._mesh.num_stages - self._mesh.stage - 1,
            self._num_microbatches,
        )
        num_steady = self._num_microbatches - num_warmup

        input_objs: list[Any] = []
        output_objs: list[Any] = []

        # Accumulate the loss on the same device the batch (and hence the
        # model output) lives on, rather than assuming CUDA whenever a GPU is
        # visible — the gloo CPU tests run the model on CPU even with a GPU
        # present.
        device = _first_tensor_device(self._batch)
        if device is None:
            device = (
                torch.device(f"cuda:{torch.cuda.current_device()}")
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else torch.device("cpu")
            )

        accum_loss: torch.Tensor | None = None
        if return_loss and self._mesh.is_last_stage():
            accum_loss = torch.zeros(1, device=device)

        outputs: list[Any] | None = (
            [] if return_outputs and self._mesh.is_last_stage() else None
        )

        # Warmup: later stages fill their activation buffers.
        for _ in range(num_warmup):
            input_obj = self._recv_forward()
            output_obj = self._forward_step(input_obj, criterion, accum_loss, outputs)
            self._send_forward(output_obj)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        # Steady state: alternate one forward, one backward.
        if num_steady > 0:
            input_obj = self._recv_forward()

        for i in range(num_steady):
            last = i == num_steady - 1
            output_obj = self._forward_step(input_obj, criterion, accum_loss, outputs)
            output_obj_grad = self._send_forward_recv_backward(
                output_obj, send_first=(self._mesh.stage % 2 == 0)
            )
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self._backward_step(optimizer, input_obj, output_obj, output_obj_grad)

            if last:
                self._send_backward(input_obj_grad)
            else:
                input_obj = self._send_backward_recv_forward(
                    input_obj_grad,
                    send_first=(self._mesh.stage % 2 == 0),
                )

        # Cooldown: earlier stages drain remaining backwards.
        for _ in range(num_warmup):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            output_obj_grad = self._recv_backward()
            input_obj_grad = self._backward_step(optimizer, input_obj, output_obj, output_obj_grad)
            self._send_backward(input_obj_grad)

        return {
            "loss": accum_loss,
            "outputs": _merge_batch(outputs) if outputs is not None else None,
        }
