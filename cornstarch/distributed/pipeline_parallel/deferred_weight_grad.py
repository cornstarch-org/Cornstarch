"""Deferred linear weight-gradient computation for zero-bubble schedules.

The custom autograd functions compute activation gradients during ``B`` and
record just enough state for a later ``W`` operation. Outside an active capture
they behave like ordinary linear autograd, which keeps instrumented modules safe
for diagnostics and non-scheduled backward calls.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _Capture:
    store: "DeferredWeightGradStore"
    microbatch: int


_ACTIVE_CAPTURE: ContextVar[_Capture | None] = ContextVar(
    "cornstarch_deferred_weight_grad_capture", default=None
)


def _linear_weight_grad(
    input: torch.Tensor, grad_output: torch.Tensor
) -> torch.Tensor:
    input_2d = input.reshape(-1, input.shape[-1])
    grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
    return grad_output_2d.transpose(0, 1).matmul(input_2d)


def _linear_bias_grad(grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)


def _accumulate_grad(parameter: torch.Tensor, grad: torch.Tensor) -> None:
    grad = grad.detach()
    if parameter.grad is None:
        parameter.grad = grad
    else:
        parameter.grad.add_(grad)


class DeferredWeightGradStore:
    """FIFO of per-microbatch linear weight-gradient work."""

    def __init__(self) -> None:
        self._microbatch_work: dict[int, list[Callable[[], None]]] = {}
        self._pending: deque[tuple[int, list[Callable[[], None]]]] = deque()

    @contextmanager
    def forward(self, microbatch: int) -> Iterator[None]:
        """Associate linear autograd nodes created by F with a microbatch."""
        if microbatch in self._microbatch_work:
            raise RuntimeError(f"Microbatch {microbatch} is already being captured.")
        self._microbatch_work[microbatch] = []
        token = _ACTIVE_CAPTURE.set(_Capture(self, microbatch))
        try:
            yield
        finally:
            _ACTIVE_CAPTURE.reset(token)

    @contextmanager
    def backward(self, microbatch: int) -> Iterator[None]:
        """Run B, then expose its collected weight work to the W FIFO."""
        if microbatch not in self._microbatch_work:
            raise RuntimeError(
                f"Microbatch {microbatch} has no matching deferred forward capture."
            )
        try:
            yield
        finally:
            work = self._microbatch_work.pop(microbatch)
            self._pending.append((microbatch, work))

    @contextmanager
    def capture(self, microbatch: int) -> Iterator[None]:
        """Convenience context for tests that execute F and B together."""
        with self.forward(microbatch):
            yield
        work = self._microbatch_work.pop(microbatch)
        self._pending.append((microbatch, work))

    def defer(self, microbatch: int, work: Callable[[], None]) -> None:
        try:
            self._microbatch_work[microbatch].append(work)
        except KeyError as error:
            raise RuntimeError(
                f"No active deferred-weight capture for microbatch {microbatch}."
            ) from error

    def execute(self, microbatch: int) -> None:
        if not self._pending:
            raise RuntimeError("Deferred W queue is empty.")
        pending_microbatch, work = self._pending.popleft()
        if pending_microbatch != microbatch:
            raise RuntimeError(
                "Deferred W FIFO mismatch: expected microbatch "
                f"{microbatch}, found {pending_microbatch}."
            )
        with torch.no_grad():
            for item in work:
                item()
        work.clear()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def assert_empty(self) -> None:
        if self._microbatch_work or self._pending:
            raise RuntimeError("Deferred weight-gradient state leaked across steps.")


class _DeferredLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight)
        ctx.weight_target = weight
        ctx.has_bias = bias is not None
        ctx.capture = _ACTIVE_CAPTURE.get()
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight = ctx.saved_tensors
        grad_input = (
            grad_output.matmul(weight)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_bias = (
            _linear_bias_grad(grad_output)
            if ctx.has_bias and ctx.needs_input_grad[2]
            else None
        )
        grad_weight = None
        if ctx.needs_input_grad[1]:
            capture = ctx.capture
            if capture is None:
                grad_weight = _linear_weight_grad(input, grad_output)
            else:
                weight_target = ctx.weight_target
                capture.store.defer(
                    capture.microbatch,
                    lambda input=input, grad_output=grad_output, weight=weight_target: (
                        _accumulate_grad(
                            weight, _linear_weight_grad(input, grad_output)
                        )
                    ),
                )
        return grad_input, grad_weight, grad_bias


class _DeferredExpertLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        expert_index: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weights)
        ctx.weights_target = weights
        ctx.expert_index = expert_index
        ctx.capture = _ACTIVE_CAPTURE.get()
        return F.linear(input, weights[expert_index])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weights = ctx.saved_tensors
        expert_index = ctx.expert_index
        weight = weights[expert_index]
        grad_input = (
            grad_output.matmul(weight)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weights = None
        if ctx.needs_input_grad[1]:
            capture = ctx.capture
            if capture is None:
                grad_weights = torch.zeros_like(weights)
                grad_weights[expert_index].copy_(
                    _linear_weight_grad(input, grad_output)
                )
            else:
                weights_target = ctx.weights_target

                def accumulate() -> None:
                    grad = torch.zeros_like(weights_target)
                    grad[expert_index].copy_(
                        _linear_weight_grad(input, grad_output)
                    )
                    _accumulate_grad(weights_target, grad)

                capture.store.defer(capture.microbatch, accumulate)
        return grad_input, grad_weights, None


def deferred_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear whose weight gradient can be split from activation backward."""
    return _DeferredLinear.apply(input, weight, bias)


def deferred_expert_linear(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_index: int | torch.Tensor,
) -> torch.Tensor:
    """Deferred linear over one slice of a stacked expert parameter."""
    if isinstance(expert_index, torch.Tensor):
        expert_index = int(expert_index.item())
    return _DeferredExpertLinear.apply(input, weights, expert_index)


def _deferred_linear_forward(module: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    return deferred_linear(input, module.weight, module.bias)


def apply_deferred_weight_gradients(module: nn.Module) -> None:
    """Instrument every ordinary linear module for split B/W execution.

    Instance-level forward replacement preserves module identity, parameters,
    state-dict names, hooks installed by DTensor TP, and tied parameter objects.
    """
    for child in module.modules():
        if not isinstance(child, nn.Linear):
            continue
        if getattr(child, "_cornstarch_deferred_weight_grad", False):
            continue
        child.forward = MethodType(_deferred_linear_forward, child)
        child._cornstarch_deferred_weight_grad = True
