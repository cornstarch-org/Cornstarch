from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RepeatedLayerOffloadConfig:
    """Configuration for repeated-layer CPU offload.

    This object is intentionally small because users should only decide whether
    repeated layers live on CPU and which CUDA device executes them. Everything
    else is runtime policy: Cornstarch keeps the optimizer-visible repeated
    layer parameters on pinned CPU storage, creates short-lived CUDA copies for
    execution, stores forward activations on CPU, and recomputes layers during
    backward so gradients can be returned to those CPU master parameters.
    """

    enabled: bool = False
    execution_device: str | torch.device = "cuda"
    cpu_device: str | torch.device = "cpu"

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
    manager: _RepeatedLayerOffloadRuntime | None = None,
) -> torch.Tensor:
    """Run a repeated layer stack with CPU masters and transient device copies.

    This is the only offload-aware execution loop used by Cornstarch model
    wrappers. It keeps model classes simple: they own module structure and
    forward specs, while this function owns layer prefetch, activation movement,
    device-copy lifetime, and the custom autograd boundary that recomputes each
    layer during backward.
    """
    owns_manager = manager is None
    manager = manager or create_repeated_layer_offload_runtime(layers, config)
    retain_forward_layers_for_backward = activation_checkpoint_recompute_active()

    try:
        for layer_idx, layer in enumerate(layers):
            if spec.should_skip_layer(model, layer_idx, context, **loop_kwargs):
                continue

            next_layer_idx = _next_executable_layer_index(
                model, layers, spec, layer_idx, context, loop_kwargs
            )

            layer_kwargs = spec.get_layer_kwargs(
                model, layer_idx, context, **loop_kwargs
            )
            hidden_states = _OffloadedRepeatedLayer.apply(
                hidden_states,
                _LayerExecutionRequest(
                    model=model,
                    spec=spec,
                    layer_idx=layer_idx,
                    next_layer_idx=next_layer_idx,
                    layer_kwargs=layer_kwargs,
                    context=context,
                    loop_kwargs=loop_kwargs,
                    manager=manager,
                ),
                *tuple(layer.parameters(recurse=True)),
            )
            if not retain_forward_layers_for_backward:
                manager.free(layer_idx, direction="forward")

        return _move_to_device(hidden_states, manager.execution_device)
    finally:
        if owns_manager and not retain_forward_layers_for_backward:
            manager.free_all()


def create_repeated_layer_offload_runtime(
    layers: nn.ModuleList, config: RepeatedLayerOffloadConfig
) -> _RepeatedLayerOffloadRuntime:
    """Create a runtime manager so callers can prefetch before layer iteration."""
    return _RepeatedLayerOffloadRuntime(layers, config)


class _RepeatedLayerOffloadRuntime:
    """Owns the live offload state for one repeated-layer traversal.

    The model still owns the real repeated layers. This runtime only manages
    temporary execution resources: CUDA streams, prefetched layer copies, and
    pinned CPU storage used for activations and gradients. A prefetched layer is
    stored as ``(module, ready_event, device_flat_tensors)`` instead of a small
    wrapper class: the tuple is local to this runtime, and the field meanings
    are documented at the dictionary declaration below.

    The important invariant is that optimizer-visible parameters stay attached
    to the original CPU modules. CUDA modules are disposable views into flat CUDA
    tensors, so freeing a prefetched entry releases the transient layer without
    changing optimizer state.
    """

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
        self._prefetch_stream = torch.cuda.Stream(device=self.execution_device)
        self._activation_stream = torch.cuda.Stream(device=self.execution_device)
        self._gradient_stream = torch.cuda.Stream(device=self.execution_device)
        self._device_layers: dict[
            int,
            tuple[nn.Module, torch.cuda.Event, tuple[torch.Tensor, ...]],
        ] = {}

    def prefetch(self, layer_idx: int, direction: str) -> nn.Module:
        """Create or return a transient execution-device copy for a layer."""
        if layer_idx in self._device_layers:
            return self._device_layers[layer_idx][0]

        cpu_layer = self.layers[layer_idx]
        self._ensure_cpu_master(cpu_layer)
        device_layer = copy.deepcopy(_get_or_create_meta_layer_template(cpu_layer))
        device_layer.train(cpu_layer.training)

        with torch.cuda.stream(self._prefetch_stream):
            device_flat_tensors = self._copy_flat_module_tensors_to_device(
                layer_idx, cpu_layer, device_layer
            )
            ready_event = torch.cuda.Event()
            ready_event.record(self._prefetch_stream)

        # Keep the flat CUDA tensors alive because module parameters/buffers are
        # views into them. Dropping them would invalidate the transient module.
        self._device_layers[layer_idx] = (
            device_layer,
            ready_event,
            tuple(device_flat_tensors),
        )
        return device_layer

    def take(self, layer_idx: int, direction: str) -> nn.Module:
        """Return an execution copy, creating it if it was not prefetched."""
        self.prefetch(layer_idx, direction=direction)
        return self._wait_for_prefetched_layer(layer_idx)

    def free(self, layer_idx: int, direction: str) -> None:
        """Release a transient execution-device copy for a layer."""
        self._device_layers.pop(layer_idx, None)

    def free_all(self) -> None:
        """Release every transient layer copy owned by this runtime."""
        for layer_idx in list(self._device_layers):
            self.free(layer_idx, direction="cleanup")

    def _ensure_cpu_master(self, layer: nn.Module) -> None:
        tensors = list(layer.parameters(recurse=True)) + list(layer.buffers(recurse=True))
        if any(tensor.is_meta for tensor in tensors):
            raise RuntimeError("Cannot offload repeated layers before materialize().")
        if any(tensor.device != self.cpu_device for tensor in tensors):
            layer.to(self.cpu_device)
        _get_or_create_meta_layer_template(layer)
        self._pin_cpu_master_tensors(layer)

    def _pin_cpu_master_tensors(self, layer: nn.Module) -> None:
        """Keep optimizer-visible CPU masters in pinned storage for fast refreshes."""
        _get_or_create_flat_cpu_groups(layer, kind="parameter")
        _get_or_create_flat_cpu_groups(layer, kind="buffer")

    def _wait_for_prefetched_layer(self, layer_idx: int) -> nn.Module:
        device_layer, ready_event, _device_flat_tensors = self._device_layers[layer_idx]
        torch.cuda.current_stream(self.execution_device).wait_event(
            ready_event
        )
        return device_layer

    def _copy_flat_module_tensors_to_device(
        self, layer_idx: int, cpu_layer: nn.Module, device_layer: nn.Module
    ) -> list[torch.Tensor]:
        """Transfer layer tensors as flat CUDA groups and install module views."""
        device_flat_tensors: list[torch.Tensor] = []
        with torch.no_grad():
            parameter_groups = _get_or_create_flat_cpu_groups(cpu_layer, kind="parameter")
            if parameter_groups:
                device_flat_tensors.extend(
                    self._copy_flat_groups_to_device(
                        layer_idx,
                        parameter_groups,
                        device_layer,
                    )
                )

            buffer_groups = _get_or_create_flat_cpu_groups(cpu_layer, kind="buffer")
            if buffer_groups:
                device_flat_tensors.extend(
                    self._copy_flat_groups_to_device(
                        layer_idx,
                        buffer_groups,
                        device_layer,
                    )
                )
        return device_flat_tensors

    def _copy_flat_groups_to_device(
        self,
        layer_idx: int,
        groups: list[_FlatTensorGroup],
        device_layer: nn.Module,
    ) -> list[torch.Tensor]:
        device_flat_tensors: list[torch.Tensor] = []
        for group in groups:
            if not group.cpu_tensor.is_pinned():
                raise RuntimeError("Offloaded layer flat CPU transfer source must be pinned.")

            device_flat = group.cpu_tensor.to(
                device=self.execution_device,
                non_blocking=True,
            )
            device_flat_tensors.append(device_flat)
            _install_flat_group_views(device_layer, device_flat, group)

        return device_flat_tensors

    def copy_activation_to_cpu(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Queue activation D2H copy on the reusable activation stream."""
        return _copy_tensor_flat_to_cpu_async(
            tensor,
            self.cpu_device,
            self._activation_stream,
        )

    def move_gradient_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Queue activation-gradient D2H copy on the reusable gradient stream."""
        return _move_tensor_flat_to_device(
            tensor,
            self.cpu_device,
            copy_stream=self._gradient_stream,
        )

    def move_parameter_grads_to_cpu(
        self,
        parameter_grads: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, ...]:
        """Queue parameter-gradient D2H copies on the reusable gradient stream."""
        return _move_parameter_grads_flat_to_cpu(
            parameter_grads,
            self.cpu_device,
            self._gradient_stream,
        )


@dataclass(frozen=True)
class _TensorSliceMetadata:
    """Describes one parameter or buffer inside a flattened transfer tensor.

    Each repeated layer is copied by dtype groups rather than by individual
    parameters. This metadata is the map back from a flat tensor slice to the
    original module path, shape, and trainability. It is shared by CPU master
    storage and transient CUDA storage so both sides install identical views.
    """

    name: str
    shape: torch.Size
    dtype: torch.dtype
    numel: int
    offset: int
    requires_grad: bool
    kind: str


@dataclass(frozen=True)
class _FlatTensorGroup:
    """Pinned CPU backing storage for same-dtype layer tensors.

    The CPU tensor is the long-lived master storage used by the original module.
    Module parameters keep their existing ``Parameter`` objects but point their
    ``.data`` at slices of this tensor, which preserves optimizer references.
    During prefetch, the full flat tensor is copied to CUDA once and transient
    CUDA parameters/buffers become views into the copied flat tensor.
    """

    cpu_tensor: torch.Tensor
    metadata: tuple[_TensorSliceMetadata, ...]


@dataclass(frozen=True)
class _LayerExecutionRequest:
    """Non-tensor state carried through the custom autograd boundary.

    ``torch.autograd.Function`` accepts tensor arguments for gradient plumbing
    plus opaque Python objects for execution context. This request bundles the
    model, forward spec, layer index, kwargs, and runtime so forward and backward
    do not rely on positional tuples with many entries. The CPU layer itself is
    deliberately not stored here; the original parameters are passed as tensor
    arguments to ``apply`` so autograd can return their CPU gradients.
    """

    model: nn.Module
    spec: Any
    layer_idx: int
    next_layer_idx: int | None
    layer_kwargs: dict[str, Any]
    context: dict[str, Any]
    loop_kwargs: dict[str, Any]
    manager: _RepeatedLayerOffloadRuntime


class _OffloadedRepeatedLayer(torch.autograd.Function):
    """Custom autograd node for one offloaded repeated layer.

    Forward uses a transient CUDA copy of the current layer, launches the next
    layer prefetch before current compute, and stores a CPU copy of the input
    activation for backward. Backward takes a fresh CUDA copy of the same layer,
    launches the previous-layer prefetch before recompute, reconstructs the
    forward for gradient calculation, and returns CPU gradients for the original
    layer parameters.

    This class is the only place that intentionally recomputes layer work. The
    surrounding runtime owns memory movement and lifetime; this autograd node
    owns the contract between CUDA recompute and CPU master gradients.
    """

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
        input_cpu, input_cpu_ready_event = request.manager.copy_activation_to_cpu(
            hidden_states.detach()
        )
        ctx.input_cpu_ready_event = input_cpu_ready_event
        ctx.save_for_backward(input_cpu)

        device_layer = request.manager.take(request.layer_idx, direction="forward")
        if request.next_layer_idx is not None:
            request.manager.prefetch(request.next_layer_idx, direction="forward")
        input_device = _move_tensor_flat_to_device(
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
        return processed

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        request: _LayerExecutionRequest = ctx.request
        (input_cpu,) = ctx.saved_tensors
        device_layer = request.manager.take(request.layer_idx, direction="backward")
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
        input_ready_event = getattr(ctx, "input_cpu_ready_event", None)
        if input_ready_event is not None:
            torch.cuda.current_stream(request.manager.execution_device).wait_event(
                input_ready_event
            )
        input_device = _move_tensor_flat_to_device(
            input_cpu, request.manager.execution_device
        )
        input_device = input_device.detach().requires_grad_(True)
        layer_kwargs = _move_to_device(request.layer_kwargs, request.manager.execution_device)
        grad_output_device = _move_tensor_flat_to_device(
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
        request.manager.free(request.layer_idx, direction="backward")

        input_grad_return = (
            None
            if input_grad is None or not ctx.input_requires_grad
            else _move_tensor_flat_to_device(input_grad.detach(), ctx.input_device)
        )
        cpu_parameter_grads = request.manager.move_parameter_grads_to_cpu(parameter_grads)
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


def activation_checkpoint_recompute_active() -> bool:
    """Return whether the current forward is being replayed inside backward.

    PyTorch activation checkpointing rebuilds the forward graph while an
    autograd graph task is already running. In that path, keeping transient CUDA
    layer copies until their custom backward nodes run lets backward reuse the
    parameters from the recompute instead of immediately offloading them and
    reloading them again.
    """
    return torch.is_grad_enabled() and torch._C._current_graph_task_id() != -1


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


def _build_flat_cpu_groups(
    named_tensors: Any,
    *,
    kind: str,
    pin_memory: bool,
) -> list[_FlatTensorGroup]:
    """Flatten named CPU tensors into pinned same-dtype groups."""
    grouped_tensors: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
    for name, tensor in named_tensors:
        if tensor.numel() == 0:
            continue
        if tensor.device.type != "cpu":
            raise RuntimeError(f"Offloaded {kind} tensors must live on CPU.")
        grouped_tensors.setdefault(tensor.dtype, []).append((name, tensor))

    flat_groups: list[_FlatTensorGroup] = []
    for dtype, tensors in grouped_tensors.items():
        total_numel = sum(tensor.numel() for _, tensor in tensors)
        flat_cpu = torch.empty(
            total_numel,
            dtype=dtype,
            device="cpu",
            pin_memory=pin_memory,
        )
        metadata: list[_TensorSliceMetadata] = []
        offset = 0
        for name, tensor in tensors:
            numel = tensor.numel()
            flat_cpu.narrow(0, offset, numel).copy_(tensor.reshape(-1))
            metadata.append(
                _TensorSliceMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    numel=numel,
                    offset=offset,
                    requires_grad=bool(getattr(tensor, "requires_grad", False)),
                    kind=kind,
                )
            )
            offset += numel
        flat_groups.append(_FlatTensorGroup(cpu_tensor=flat_cpu, metadata=tuple(metadata)))

    return flat_groups


def _get_or_create_flat_cpu_groups(
    layer: nn.Module, *, kind: str
) -> list[_FlatTensorGroup]:
    """Return persistent pinned flat CPU storage backing layer tensors."""
    attr_name = f"_cornstarch_offload_flat_{kind}_groups"
    cached_groups = getattr(layer, attr_name, None)
    if cached_groups is not None:
        return cached_groups

    named_tensors = (
        list(layer.named_parameters(recurse=True))
        if kind == "parameter"
        else list(layer.named_buffers(recurse=True))
    )
    groups = _build_flat_cpu_groups(
        named_tensors,
        kind=kind,
        pin_memory=True,
    )
    _install_flat_cpu_group_views(layer, groups, kind=kind)
    setattr(layer, attr_name, groups)
    return groups


def _install_flat_cpu_group_views(
    layer: nn.Module, groups: list[_FlatTensorGroup], *, kind: str
) -> None:
    """Point CPU layer tensors at their persistent flat pinned backing storage."""
    with torch.no_grad():
        if kind == "parameter":
            parameters = dict(layer.named_parameters(recurse=True))
            for group in groups:
                for metadata in group.metadata:
                    parameter = parameters[metadata.name]
                    flat_slice = group.cpu_tensor.narrow(
                        0,
                        metadata.offset,
                        metadata.numel,
                    )
                    parameter.data = flat_slice.view(metadata.shape)
            return

        for group in groups:
            _install_flat_group_views(layer, group.cpu_tensor, group)


def _install_flat_group_views(
    layer: nn.Module,
    flat_tensor: torch.Tensor,
    group: _FlatTensorGroup,
) -> None:
    """Point layer tensors at slices of one flat storage tensor."""
    for metadata in group.metadata:
        flat_slice = flat_tensor.narrow(
            0,
            metadata.offset,
            metadata.numel,
        ).view(metadata.shape)
        if metadata.kind == "parameter":
            _set_module_parameter(
                layer,
                metadata.name,
                flat_slice,
                requires_grad=metadata.requires_grad,
            )
        else:
            _set_module_buffer(layer, metadata.name, flat_slice)


def _get_or_create_meta_layer_template(layer: nn.Module) -> nn.Module:
    """Cache a meta-device module copy so prefetch avoids CPU parameter copies."""
    attr_name = "_cornstarch_offload_meta_template"
    template = layer.__dict__.get(attr_name)
    if template is not None:
        return template

    template = copy.deepcopy(layer)
    for copied_attr_name in (
        "_cornstarch_offload_flat_parameter_groups",
        "_cornstarch_offload_flat_buffer_groups",
    ):
        if hasattr(template, copied_attr_name):
            delattr(template, copied_attr_name)
    template.to_empty(device=torch.device("meta"))
    layer.__dict__[attr_name] = template
    return template


def _move_tensor_flat_to_device(
    tensor: torch.Tensor,
    device: torch.device,
    copy_stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Move a tensor as one flat transfer and restore its original view shape."""
    flat_source = tensor.detach().contiguous().view(-1)
    if device.type == "cpu" and tensor.device.type == "cuda":
        flat_dest = _copy_flat_device_tensor_to_pinned_cpu(
            flat_source,
            device,
            copy_stream,
        )[0]
    else:
        flat_dest = flat_source.to(device=device, non_blocking=True)
    return flat_dest.view(tensor.shape)


def _copy_tensor_flat_to_cpu_async(
    tensor: torch.Tensor,
    cpu_device: torch.device,
    copy_stream: torch.cuda.Stream,
) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Copy a tensor to pinned CPU storage without waiting on the host thread."""
    flat_cpu, ready_event = _copy_flat_device_tensor_to_pinned_cpu(
        tensor.detach().contiguous().view(-1),
        cpu_device,
        copy_stream,
    )
    return flat_cpu.view(tensor.shape), ready_event


def _move_parameter_grads_flat_to_cpu(
    parameter_grads: list[torch.Tensor | None],
    cpu_device: torch.device,
    copy_stream: torch.cuda.Stream,
) -> tuple[torch.Tensor | None, ...]:
    """Move parameter gradients to CPU through flat same-dtype transfers."""
    grouped_grads: dict[torch.dtype, list[tuple[int, torch.Tensor]]] = {}
    for index, grad in enumerate(parameter_grads):
        if grad is None:
            continue
        grouped_grads.setdefault(grad.dtype, []).append((index, grad.detach()))

    cpu_grads: list[torch.Tensor | None] = [None] * len(parameter_grads)
    for dtype, grads in grouped_grads.items():
        flat_device = torch.cat(
            [grad.contiguous().view(-1) for _, grad in grads],
        )
        flat_cpu = _copy_flat_device_tensor_to_pinned_cpu(
            flat_device,
            cpu_device,
            copy_stream,
        )[0]

        offset = 0
        for index, grad in grads:
            numel = grad.numel()
            cpu_grads[index] = flat_cpu.narrow(0, offset, numel).view(grad.shape)
            offset += numel

    return tuple(cpu_grads)


def _copy_flat_device_tensor_to_pinned_cpu(
    flat_source: torch.Tensor,
    cpu_device: torch.device,
    copy_stream: torch.cuda.Stream | None,
) -> tuple[torch.Tensor, torch.cuda.Event]:
    """Queue a flat D2H copy without blocking the Python thread.

    The copy runs on a side stream after the current stream's producer work and
    returns the event that marks the CPU tensor ready. Later GPU consumers can
    wait on that event without forcing a host-side ``synchronize()``.
    """
    flat_dest = torch.empty(
        flat_source.shape,
        dtype=flat_source.dtype,
        device=cpu_device,
        pin_memory=True,
    )
    device = flat_source.device
    current_stream = torch.cuda.current_stream(device)
    copy_stream = copy_stream or current_stream
    copy_stream.wait_stream(current_stream)
    with torch.cuda.stream(copy_stream):
        flat_dest.copy_(flat_source, non_blocking=True)
        ready_event = torch.cuda.Event()
        ready_event.record(copy_stream)
    flat_source.record_stream(copy_stream)
    return flat_dest, ready_event


def _set_module_buffer(root: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Replace a dotted buffer path without changing module ownership."""
    module_name, _, buffer_name = name.rpartition(".")
    module = root.get_submodule(module_name) if module_name else root
    module._buffers[buffer_name] = tensor


def _set_module_parameter(
    root: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    requires_grad: bool,
) -> None:
    """Replace a dotted parameter path without changing module ownership."""
    module_name, _, parameter_name = name.rpartition(".")
    module = root.get_submodule(module_name) if module_name else root
    module._parameters[parameter_name] = nn.Parameter(
        tensor,
        requires_grad=requires_grad,
    )
