from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn


LayerOffloadEventCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class RepeatedLayerOffloadConfig:
    """Configuration for repeated-layer CPU offload.

    The offload runtime treats each registered repeated layer as the CPU master
    copy. Forward creates short-lived GPU execution-device copies, keeps
    produced activations on CPU between repeated layers, and reconstructs GPU
    copies in reverse order during backward so gradients can be accumulated back
    onto the CPU master parameters.

    ``event_callback`` is intentionally lightweight and primarily exists for
    deterministic tests and future instrumentation. It receives an event name
    and layer index whenever the runtime prefetches, runs, frees, or records an
    activation for a repeated layer.
    """

    enabled: bool = False
    execution_device: str | torch.device = "cuda"
    cpu_device: str | torch.device = "cpu"
    event_callback: LayerOffloadEventCallback | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        if self.execution_torch_device.type != "cuda":
            raise ValueError("Repeated layer CPU offload requires a CUDA execution device.")
        if self.cpu_torch_device.type != "cpu":
            raise ValueError("Repeated layer CPU offload requires a CPU master device.")

    @property
    def execution_torch_device(self) -> torch.device:
        return torch.device(self.execution_device)

    @property
    def cpu_torch_device(self) -> torch.device:
        return torch.device(self.cpu_device)


def layer_offload_enabled(config: RepeatedLayerOffloadConfig | None) -> bool:
    """Return whether a model-level offload config requests layer offload."""
    return config is not None and config.enabled


def run_repeated_layers_with_offload(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: Any,
    hidden_states: torch.Tensor,
    context: dict[str, Any],
    loop_kwargs: dict[str, Any],
    config: RepeatedLayerOffloadConfig,
) -> torch.Tensor:
    """Run a repeated layer stack with CPU masters and transient device copies.

    This is the only offload-aware execution loop used by Cornstarch model
    wrappers. It keeps model classes simple: they own module structure and
    forward specs, while this function owns layer prefetch, activation movement,
    device-copy lifetime, and the custom autograd boundary that recomputes each
    layer during backward.
    """
    manager = _RepeatedLayerOffloadRuntime(layers, config)

    try:
        for layer_idx, layer in enumerate(layers):
            if spec.should_skip_layer(model, layer_idx, context, **loop_kwargs):
                continue

            next_layer_idx = _next_executable_layer_index(
                model, layers, spec, layer_idx, context, loop_kwargs
            )
            if next_layer_idx is not None:
                manager.prefetch(next_layer_idx, direction="forward")

            layer_kwargs = spec.get_layer_kwargs(
                model, layer_idx, context, **loop_kwargs
            )
            hidden_states = _OffloadedRepeatedLayer.apply(
                hidden_states,
                _LayerExecutionRequest(
                    model=model,
                    spec=spec,
                    layer_idx=layer_idx,
                    layer=layer,
                    layer_kwargs=layer_kwargs,
                    context=context,
                    loop_kwargs=loop_kwargs,
                    manager=manager,
                ),
                *tuple(layer.parameters(recurse=True)),
            )
            manager.free(layer_idx, direction="forward")

        return _move_to_device(hidden_states, manager.execution_device)
    finally:
        manager.free_all()


class _RepeatedLayerOffloadRuntime:
    """Stateful manager for transient repeated-layer execution copies."""

    def __init__(
        self, layers: nn.ModuleList, config: RepeatedLayerOffloadConfig
    ) -> None:
        self.layers = layers
        self.config = config
        self.execution_device = config.execution_torch_device
        self.cpu_device = config.cpu_torch_device
        if self.execution_device.type != "cuda":
            raise RuntimeError("Repeated layer CPU offload requires a CUDA execution device.")
        if self.cpu_device.type != "cpu":
            raise RuntimeError("Repeated layer CPU offload requires a CPU master device.")
        if not torch.cuda.is_available():
            raise RuntimeError("Repeated layer CPU offload requires CUDA for execution_device='cuda'.")
        self._device_layers: dict[int, nn.Module] = {}

    def prefetch(self, layer_idx: int, direction: str) -> nn.Module:
        """Create or return a transient execution-device copy for a layer."""
        if layer_idx in self._device_layers:
            return self._device_layers[layer_idx]

        cpu_layer = self.layers[layer_idx]
        self._ensure_cpu_master(cpu_layer)
        device_layer = copy.deepcopy(cpu_layer)
        device_layer.train(cpu_layer.training)
        device_layer.to(self.execution_device)
        self._device_layers[layer_idx] = device_layer
        self._record(f"prefetch_{direction}", layer_idx)
        return device_layer

    def take(self, layer_idx: int, direction: str) -> nn.Module:
        """Return an execution copy, creating it if it was not prefetched."""
        return self.prefetch(layer_idx, direction=direction)

    def free(self, layer_idx: int, direction: str) -> None:
        """Release a transient execution-device copy for a layer."""
        if self._device_layers.pop(layer_idx, None) is not None:
            self._record(f"free_{direction}", layer_idx)
            if self.execution_device.type == "cuda":
                torch.cuda.empty_cache()

    def free_all(self) -> None:
        """Release every transient layer copy owned by this runtime."""
        for layer_idx in list(self._device_layers):
            self.free(layer_idx, direction="cleanup")

    def record_activation(self, layer_idx: int) -> None:
        """Record that a layer output activation has been moved to CPU."""
        self._record("activation_cpu", layer_idx)

    def _ensure_cpu_master(self, layer: nn.Module) -> None:
        tensors = list(layer.parameters(recurse=True)) + list(layer.buffers(recurse=True))
        if any(tensor.is_meta for tensor in tensors):
            raise RuntimeError("Cannot offload repeated layers before materialize().")
        if any(tensor.device != self.cpu_device for tensor in tensors):
            layer.to(self.cpu_device)

    def _record(self, event: str, layer_idx: int) -> None:
        if self.config.event_callback is not None:
            self.config.event_callback(event, layer_idx)


@dataclass(frozen=True)
class _LayerExecutionRequest:
    """Non-tensor execution state carried through the autograd boundary."""

    model: nn.Module
    spec: Any
    layer_idx: int
    layer: nn.Module
    layer_kwargs: dict[str, Any]
    context: dict[str, Any]
    loop_kwargs: dict[str, Any]
    manager: _RepeatedLayerOffloadRuntime


class _OffloadedRepeatedLayer(torch.autograd.Function):
    """Autograd boundary that frees forward copies and recomputes in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: torch.Tensor,
        request: _LayerExecutionRequest,
        *cpu_parameters: torch.Tensor,
    ) -> torch.Tensor:
        ctx.request = request
        ctx.cpu_parameters = cpu_parameters
        ctx.input_requires_grad = hidden_states.requires_grad
        ctx.input_device = hidden_states.device
        input_cpu = _move_to_device(hidden_states.detach(), request.manager.cpu_device)
        ctx.save_for_backward(input_cpu)

        device_layer = request.manager.take(request.layer_idx, direction="forward")
        input_device = _move_to_device(
            hidden_states.detach(), request.manager.execution_device
        )
        layer_kwargs = _move_to_device(request.layer_kwargs, request.manager.execution_device)

        layer_output = device_layer(input_device, **layer_kwargs)
        processed = request.spec.process_layer_output(
            request.model,
            request.layer_idx,
            layer_output,
            request.context,
            **request.loop_kwargs,
        )
        output_cpu = _move_to_device(processed.detach(), request.manager.cpu_device)
        request.manager.record_activation(request.layer_idx)
        request.manager._record("run_forward", request.layer_idx)
        return output_cpu

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        request: _LayerExecutionRequest = ctx.request
        (input_cpu,) = ctx.saved_tensors
        prev_layer_idx = _previous_executable_layer_index(
            request.model,
            request.manager.layers,
            request.spec,
            request.layer_idx,
            request.context,
            request.loop_kwargs,
        )
        if prev_layer_idx is not None:
            request.manager.prefetch(prev_layer_idx, direction="backward")

        device_layer = request.manager.take(request.layer_idx, direction="backward")
        input_device = _move_to_device(input_cpu, request.manager.execution_device)
        input_device = input_device.detach().requires_grad_(True)
        layer_kwargs = _move_to_device(request.layer_kwargs, request.manager.execution_device)
        grad_output_device = _move_to_device(
            grad_output, request.manager.execution_device
        )

        with torch.enable_grad():
            layer_output = device_layer(input_device, **layer_kwargs)
            processed = request.spec.process_layer_output(
                request.model,
                request.layer_idx,
                layer_output,
                dict(request.context),
                **request.loop_kwargs,
            )
            device_parameters = list(device_layer.parameters(recurse=True))
            trainable_parameter_indices = [
                index
                for index, parameter in enumerate(device_parameters)
                if parameter.requires_grad
            ]
            grad_targets = [
                input_device,
                *[
                    device_parameters[index]
                    for index in trainable_parameter_indices
                ],
            ]
            gradients = torch.autograd.grad(
                processed,
                grad_targets,
                grad_output_device,
                allow_unused=True,
            )

        input_grad = gradients[0]
        trainable_parameter_grads = gradients[1:]
        parameter_grads: list[torch.Tensor | None] = [None] * len(device_parameters)
        for index, grad in zip(
            trainable_parameter_indices, trainable_parameter_grads, strict=True
        ):
            parameter_grads[index] = grad
        request.manager._record("run_backward", request.layer_idx)
        request.manager.free(request.layer_idx, direction="backward")

        input_grad_cpu = (
            None
            if input_grad is None or not ctx.input_requires_grad
            else _move_to_device(input_grad.detach(), request.manager.cpu_device)
        )
        request.manager._record("grad_cpu", request.layer_idx)
        input_grad_return = (
            None
            if input_grad_cpu is None
            else _move_to_device(input_grad_cpu, ctx.input_device)
        )
        cpu_parameter_grads = tuple(
            None
            if grad is None
            else _move_to_device(grad.detach(), request.manager.cpu_device)
            for grad in parameter_grads
        )
        return (input_grad_return, None, *cpu_parameter_grads)


def _next_executable_layer_index(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: Any,
    layer_idx: int,
    context: dict[str, Any],
    loop_kwargs: dict[str, Any],
) -> int | None:
    for next_idx in range(layer_idx + 1, len(layers)):
        if not spec.should_skip_layer(model, next_idx, context, **loop_kwargs):
            return next_idx
    return None


def _previous_executable_layer_index(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: Any,
    layer_idx: int,
    context: dict[str, Any],
    loop_kwargs: dict[str, Any],
) -> int | None:
    for previous_idx in range(layer_idx - 1, -1, -1):
        if not spec.should_skip_layer(model, previous_idx, context, **loop_kwargs):
            return previous_idx
    return None


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value
