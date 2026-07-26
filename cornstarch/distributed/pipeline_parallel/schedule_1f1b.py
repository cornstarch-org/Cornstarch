"""One-forward-one-backward pipeline-parallel training schedule."""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch.optim import Optimizer

from cornstarch.distributed.pipeline_parallel.schedule import (
    BasePipelineSchedule,
    _default_device,
    _first_tensor_device,
    _merge_batch,
)


class OneForwardOneBackwardSchedule(BasePipelineSchedule):
    """One-forward-one-backward pipeline schedule over the global pipeline."""

    def step(
        self,
        microbatches: list[dict[str, torch.Tensor]],
        criterion: Callable[..., Any],
        optimizer: Optimizer | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        if self._idle:
            # No stage in this batch's plan (e.g. encoder ranks on a text-only
            # step): nothing to run, no transfers to anyone.
            return {"loss": None, "outputs": None}

        if not isinstance(microbatches, list):
            microbatches = [microbatches]
        num_microbatches = len(microbatches)
        self._device = _first_tensor_device(microbatches) or _default_device()

        num_warmup = min(self._num_stages - self._stage - 1, num_microbatches)
        num_steady = num_microbatches - num_warmup

        accum_loss: torch.Tensor | None = None
        if return_loss and self.is_last_stage():
            accum_loss = torch.zeros(1, device=self._device)
        outputs: list[Any] | None = (
            [] if return_outputs and self.is_last_stage() else None
        )

        input_objs: list[Any] = []
        output_objs: list[Any] = []
        mb_index = 0

        # Warmup.
        for _ in range(num_warmup):
            active_microbatch = microbatches[mb_index]
            input_obj = self._recv_forward(active_microbatch)
            output_obj = self._forward_step(
                active_microbatch, input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            self._send_forward(output_obj, active_microbatch)
            input_objs.append(input_obj)
            output_objs.append(output_obj)

        if num_steady > 0:
            input_obj = self._recv_forward(microbatches[mb_index])

        # Steady state.
        for i in range(num_steady):
            last = i == num_steady - 1
            active_microbatch = microbatches[mb_index]
            output_obj = self._forward_step(
                active_microbatch, input_obj, criterion,
                num_microbatches, accum_loss, outputs,
            )
            mb_index += 1
            output_obj_grad = self._send_forward_recv_backward(
                output_obj, active_microbatch
            )
            input_objs.append(input_obj)
            output_objs.append(output_obj)

            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            input_obj_grad = self._backward_step(input_obj, output_obj, output_obj_grad)

            if last:
                self._send_backward(input_obj_grad)
            else:
                input_obj = self._send_backward_recv_forward(
                    input_obj_grad, microbatches[mb_index]
                )

        # Cooldown.
        for _ in range(num_warmup):
            input_obj = input_objs.pop(0)
            output_obj = output_objs.pop(0)
            output_obj_grad = self._recv_backward()
            input_obj_grad = self._backward_step(input_obj, output_obj, output_obj_grad)
            self._send_backward(input_obj_grad)

        return {
            "loss": accum_loss if return_loss else None,
            "outputs": _merge_batch(outputs) if outputs is not None else None,
        }
