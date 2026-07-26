"""Count-heuristic ZB-H2 pipeline-parallel training schedule."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import torch
from torch.optim import Optimizer

from cornstarch.distributed.pipeline_parallel.deferred_weight_grad import (
    DeferredWeightGradStore,
)
from cornstarch.distributed.pipeline_parallel.schedule import (
    BasePipelineSchedule,
    _default_device,
    _first_tensor_device,
    _merge_batch,
)


class OperationType(str, Enum):
    FORWARD = "F"
    BACKWARD = "B"
    WEIGHT = "W"


@dataclass(frozen=True)
class PipelineOperation:
    """One count-selected ZB-H2 operation and its microbatch index."""

    type: OperationType
    microbatch: int


def h2_warmup_for_stage(num_stages: int, stage: int) -> int:
    """Return ``u_s = 2(p-s)-1`` for a valid zero-based stage."""
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1.")
    if not 0 <= stage < num_stages:
        raise ValueError(
            f"stage must be in [0, {num_stages}), got {stage}."
        )
    return 2 * (num_stages - stage) - 1


def select_main_operations(
    forward_count: int,
    backward_count: int,
    weight_count: int,
    num_microbatches: int,
) -> tuple[PipelineOperation, ...]:
    """Select one H2 main-loop iteration purely from unconsumed counts.

    Forward and backward checks are deliberately independent. The W check sees
    the B selected in this iteration, yielding F/B/W, then B/W, then W if a
    caller supplies a state with only deferred weight work remaining.
    """
    if not 0 <= weight_count <= backward_count <= forward_count <= num_microbatches:
        raise ValueError(
            "ZB-H2 counters must satisfy 0 <= w <= b <= f <= m."
        )

    operations: list[PipelineOperation] = []
    next_backward_count = backward_count
    if forward_count < num_microbatches:
        operations.append(PipelineOperation(OperationType.FORWARD, forward_count))
    if backward_count < num_microbatches:
        operations.append(PipelineOperation(OperationType.BACKWARD, backward_count))
        next_backward_count += 1
    if weight_count < next_backward_count:
        operations.append(PipelineOperation(OperationType.WEIGHT, weight_count))
    return tuple(operations)


class ZeroBubblePipelineSchedule(BasePipelineSchedule):
    """ZB-H2 with an H2 warmup and a count-only F/B/W main program."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weight_grad_store = DeferredWeightGradStore()

    def _assert_iteration_state(
        self,
        forward_count: int,
        backward_count: int,
        weight_count: int,
        num_microbatches: int,
        input_objs: list[Any],
        output_objs: list[Any],
    ) -> None:
        assert 0 <= weight_count <= backward_count <= forward_count <= num_microbatches
        expected_activations = forward_count - backward_count
        assert len(input_objs) == expected_activations
        assert len(output_objs) == expected_activations
        assert self._weight_grad_store.pending_count == backward_count - weight_count

    def step(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        if self._idle:
            return {"loss": None, "outputs": None}

        if not isinstance(microbatches, list):
            microbatches = [microbatches]
        num_microbatches = len(microbatches)
        minimum_microbatches = 2 * self._num_stages - 1
        if num_microbatches < minimum_microbatches:
            raise ValueError(
                "ZB-H2 requires at least 2p-1 microbatches: "
                f"got m={num_microbatches}, p={self._num_stages}, "
                f"minimum={minimum_microbatches}."
            )
        self._weight_grad_store.assert_empty()
        if self._seam_states:
            raise RuntimeError("Cross-mesh seam state leaked into a ZB-H2 step.")

        self._device = _first_tensor_device(microbatches) or _default_device()
        num_warmup = h2_warmup_for_stage(self._num_stages, self._stage)

        accum_loss: torch.Tensor | None = None
        if return_loss and self.is_last_stage():
            accum_loss = torch.zeros(1, device=self._device)
        outputs: list[Any] | None = (
            [] if return_outputs and self.is_last_stage() else None
        )

        input_objs: list[Any] = []
        output_objs: list[Any] = []

        # H2 warmup: u_s forwards, with adjacent stages offset by two.
        for forward_count in range(num_warmup):
            active_microbatch = microbatches[forward_count]
            input_obj = self._recv_forward(active_microbatch)
            with self._weight_grad_store.forward(forward_count):
                output_obj = self._forward_step(
                    active_microbatch,
                    input_obj,
                    criterion,
                    num_microbatches,
                    accum_loss,
                    outputs,
                )
            self._send_forward(output_obj, active_microbatch)
            input_objs.append(input_obj)
            output_objs.append(output_obj)
            self._assert_iteration_state(
                forward_count + 1,
                0,
                0,
                num_microbatches,
                input_objs,
                output_objs,
            )

        forward_count = num_warmup
        backward_count = 0
        weight_count = 0
        self._assert_iteration_state(
            forward_count,
            backward_count,
            weight_count,
            num_microbatches,
            input_objs,
            output_objs,
        )

        next_input_obj: Any | None = None
        if forward_count < num_microbatches:
            next_input_obj = self._recv_forward(microbatches[forward_count])

        while (
            forward_count < num_microbatches
            or backward_count < num_microbatches
            or weight_count < num_microbatches
        ):
            operations = select_main_operations(
                forward_count,
                backward_count,
                weight_count,
                num_microbatches,
            )
            operation_types = {operation.type for operation in operations}
            output_obj_grad: Any | None = None

            # Operations are always executed in the selected F, B, W order.
            for operation in operations:
                if operation.type is OperationType.FORWARD:
                    active_microbatch = microbatches[operation.microbatch]
                    with self._weight_grad_store.forward(operation.microbatch):
                        output_obj = self._forward_step(
                            active_microbatch,
                            next_input_obj,
                            criterion,
                            num_microbatches,
                            accum_loss,
                            outputs,
                        )
                    if OperationType.BACKWARD in operation_types:
                        output_obj_grad = self._send_forward_recv_backward(
                            output_obj, active_microbatch
                        )
                    else:
                        self._send_forward(output_obj, active_microbatch)
                    input_objs.append(next_input_obj)
                    output_objs.append(output_obj)
                    forward_count += 1

                elif operation.type is OperationType.BACKWARD:
                    if OperationType.FORWARD not in operation_types:
                        output_obj_grad = self._recv_backward()
                    input_obj = input_objs.pop(0)
                    output_obj = output_objs.pop(0)
                    with self._weight_grad_store.backward(operation.microbatch):
                        input_obj_grad = self._backward_step(
                            input_obj, output_obj, output_obj_grad
                        )
                    backward_count += 1

                    if forward_count < num_microbatches:
                        next_input_obj = self._send_backward_recv_forward(
                            input_obj_grad, microbatches[forward_count]
                        )
                    else:
                        self._send_backward(input_obj_grad)
                        next_input_obj = None

                else:
                    self._weight_grad_store.execute(operation.microbatch)
                    weight_count += 1

            self._assert_iteration_state(
                forward_count,
                backward_count,
                weight_count,
                num_microbatches,
                input_objs,
                output_objs,
            )

        assert not input_objs
        assert not output_objs
        self._weight_grad_store.assert_empty()
        if self._seam_states:
            raise RuntimeError("Cross-mesh seam state leaked out of a ZB-H2 step.")

        return {
            "loss": accum_loss if return_loss else None,
            "outputs": _merge_batch(outputs) if outputs is not None else None,
        }
