from __future__ import annotations

import copy

import pytest
import torch

import cornstarch.models.layer_offload as layer_offload_module
from cornstarch.models import RepeatedLayerOffloadConfig, from_hf_config
from tests.model.model_configs import llama_config
from tests.model.synthetic_model_configs import (
    SyntheticLayer,
    SyntheticModel,
    synthetic_layer_stack,
)


def _event_index(events: list[tuple[str, int]], event: str, layer_idx: int) -> int:
    return events.index((event, layer_idx))


def _collect_offload_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Observe private offload scheduling from tests without production callbacks."""
    events: list[tuple[str, int]] = []
    runtime_cls = layer_offload_module._RepeatedLayerOffloadRuntime
    offloaded_layer_cls = layer_offload_module._OffloadedRepeatedLayer

    original_prefetch = runtime_cls.prefetch
    original_take = runtime_cls.take
    original_free = runtime_cls.free
    original_copy_flat_module_tensors_to_device = (
        runtime_cls._copy_flat_module_tensors_to_device
    )
    original_copy_flat_groups_to_device = runtime_cls._copy_flat_groups_to_device
    original_forward = offloaded_layer_cls.forward
    original_backward = offloaded_layer_cls.backward
    original_synthetic_layer_forward = SyntheticLayer.forward
    device_layer_indices: dict[int, int] = {}

    def prefetch(self, layer_idx: int, direction: str) -> torch.nn.Module:
        was_prefetched = layer_idx in self._device_layers
        result = original_prefetch(self, layer_idx, direction)
        if not was_prefetched:
            device_layer_indices[id(result)] = layer_idx
            events.append((f"prefetch_{direction}", layer_idx))
            events.append(("prefetch_cuda", layer_idx))
        return result

    def take(self, layer_idx: int, direction: str) -> torch.nn.Module:
        result = original_take(self, layer_idx, direction)
        events.append(("prefetch_wait", layer_idx))
        return result

    def free(self, layer_idx: int, direction: str) -> None:
        was_prefetched = layer_idx in self._device_layers
        original_free(self, layer_idx, direction)
        if was_prefetched:
            events.append((f"free_{direction}", layer_idx))

    def copy_flat_module_tensors_to_device(
        self,
        layer_idx: int,
        cpu_layer: torch.nn.Module,
        device_layer: torch.nn.Module,
    ) -> list[torch.Tensor]:
        if any(parameter.numel() for parameter in cpu_layer.parameters(recurse=True)):
            events.append(("flatten_parameters", layer_idx))
        if any(buffer.numel() for buffer in cpu_layer.buffers(recurse=True)):
            events.append(("flatten_buffers", layer_idx))
        return original_copy_flat_module_tensors_to_device(
            self,
            layer_idx,
            cpu_layer,
            device_layer,
        )

    def copy_flat_groups_to_device(
        self,
        layer_idx: int,
        groups: list[object],
        device_layer: torch.nn.Module,
    ) -> list[torch.Tensor]:
        kind = groups[0].metadata[0].kind
        event_prefix = "parameters" if kind == "parameter" else "buffers"
        result = original_copy_flat_groups_to_device(
            self,
            layer_idx,
            groups,
            device_layer,
        )
        events.append((f"pinned_flat_{event_prefix}", layer_idx))
        events.append((f"flat_{event_prefix}_transfer", layer_idx))
        events.append((f"unflatten_{event_prefix}", layer_idx))
        return result

    def forward(
        ctx: object,
        hidden_states: torch.Tensor,
        request: object,
        *cpu_parameters: torch.Tensor,
    ) -> torch.Tensor:
        output = original_forward(ctx, hidden_states, request, *cpu_parameters)
        events.append(("activation_cpu", request.layer_idx))
        events.append(("run_forward", request.layer_idx))
        return output

    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[object, ...]:
        request = ctx.request
        result = original_backward(ctx, grad_output)
        events.append(("run_backward", request.layer_idx))
        events.append(("grad_cpu", request.layer_idx))
        events.append(("flatten_grad", request.layer_idx))
        return result

    def synthetic_layer_forward(
        self: SyntheticLayer, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        layer_idx = device_layer_indices.get(id(self))
        if layer_idx is not None:
            event = (
                "layer_backward_recompute_enter"
                if torch.is_grad_enabled()
                else "layer_forward_enter"
            )
            events.append((event, layer_idx))
        return original_synthetic_layer_forward(self, hidden_states)

    monkeypatch.setattr(runtime_cls, "prefetch", prefetch)
    monkeypatch.setattr(runtime_cls, "take", take)
    monkeypatch.setattr(runtime_cls, "free", free)
    monkeypatch.setattr(
        runtime_cls,
        "_copy_flat_module_tensors_to_device",
        copy_flat_module_tensors_to_device,
    )
    monkeypatch.setattr(
        runtime_cls,
        "_copy_flat_groups_to_device",
        copy_flat_groups_to_device,
    )
    monkeypatch.setattr(offloaded_layer_cls, "forward", staticmethod(forward))
    monkeypatch.setattr(offloaded_layer_cls, "backward", staticmethod(backward))
    monkeypatch.setattr(SyntheticLayer, "forward", synthetic_layer_forward)

    return events


def test_layer_offload_config_threads_through_model_factory() -> None:
    config = RepeatedLayerOffloadConfig(enabled=True)

    model = from_hf_config(
        llama_config(),
        attn_implementation="kernels-community/flash-attn3",
        layer_offload_config=config,
    )

    assert model.layer_offload_config is config
    assert model.uses_layer_offload
    assert model.is_gradient_checkpointing


def test_repeated_layer_offload_is_disabled_by_default() -> None:
    model = from_hf_config(
        llama_config(), attn_implementation="kernels-community/flash-attn3"
    )

    assert model.layer_offload_config is None
    assert not model.uses_layer_offload
    assert model.is_gradient_checkpointing


def test_enabled_layer_offload_rejects_cpu_execution_device() -> None:
    with pytest.raises(ValueError, match="CUDA execution device"):
        RepeatedLayerOffloadConfig(enabled=True, execution_device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_forward_prefetches_next_layer_and_records_cpu_activations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _collect_offload_events(monkeypatch)
    model = SyntheticModel(
        synthetic_layer_stack(),
        RepeatedLayerOffloadConfig(enabled=True),
    )

    output = model(torch.randn(2, 4, device="cuda", requires_grad=True))

    assert output.last_hidden_state.device.type == "cuda"
    assert [event for event in events if event[0] == "activation_cpu"] == [
        ("activation_cpu", 0),
        ("activation_cpu", 1),
        ("activation_cpu", 2),
    ]
    assert _event_index(events, "prefetch_forward", 0) < _event_index(
        events, "run_forward", 0
    )
    assert _event_index(events, "prefetch_forward", 1) < _event_index(
        events, "run_forward", 0
    )
    assert _event_index(events, "prefetch_forward", 1) < _event_index(
        events, "layer_forward_enter", 0
    )
    assert _event_index(events, "prefetch_forward", 2) < _event_index(
        events, "run_forward", 1
    )
    assert _event_index(events, "prefetch_forward", 2) < _event_index(
        events, "layer_forward_enter", 1
    )
    for layer_idx in range(3):
        assert events.count(("pinned_flat_parameters", layer_idx)) == 1
        assert events.count(("flat_parameters_transfer", layer_idx)) == 1
        assert events.count(("unflatten_parameters", layer_idx)) == 1
    assert ("free_forward", 0) in events
    assert ("free_forward", 1) in events
    assert ("free_forward", 2) in events


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_backward_prefetches_previous_layer_and_accumulates_cpu_grads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _collect_offload_events(monkeypatch)
    offloaded_layers = synthetic_layer_stack()
    direct_layers = copy.deepcopy(offloaded_layers)
    offloaded_model = SyntheticModel(
        offloaded_layers,
        RepeatedLayerOffloadConfig(enabled=True),
    )
    direct_model = SyntheticModel(direct_layers).to("cuda")
    hidden_states = torch.randn(2, 4, device="cuda", requires_grad=True)
    direct_hidden_states = hidden_states.detach().clone().requires_grad_(True)

    offloaded_loss = offloaded_model(hidden_states).last_hidden_state.sum()
    direct_loss = direct_model(direct_hidden_states).last_hidden_state.sum()

    offloaded_loss.backward()
    direct_loss.backward()

    for layer_idx in range(3):
        assert events.count(("flat_parameters_transfer", layer_idx)) == 2
        assert events.count(("prefetch_backward", layer_idx)) == 0
        assert ("free_backward", layer_idx) in events
    for offloaded_param, direct_param in zip(
        offloaded_model.parameters(), direct_model.parameters(), strict=True
    ):
        assert offloaded_param.grad is not None
        assert offloaded_param.grad.device.type == "cpu"
        assert torch.allclose(offloaded_param.grad, direct_param.grad.cpu(), atol=1e-6)
    assert hidden_states.grad is not None
    assert torch.allclose(hidden_states.grad, direct_hidden_states.grad, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_execution_matches_direct_execution_after_optimizer_step() -> None:
    offloaded_model = SyntheticModel(
        synthetic_layer_stack(width=8, depth=4).to(dtype=torch.bfloat16),
        RepeatedLayerOffloadConfig(enabled=True),
    )
    direct_model = copy.deepcopy(offloaded_model)
    direct_model.layer_offload_config = None
    direct_model.to("cuda")
    hidden_states = torch.randn(
        3, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    direct_hidden_states = hidden_states.detach().clone().requires_grad_(True)
    offloaded_optimizer = torch.optim.SGD(offloaded_model.parameters(), lr=0.01)
    direct_optimizer = torch.optim.SGD(direct_model.parameters(), lr=0.01)

    offloaded_output = offloaded_model(hidden_states).last_hidden_state
    direct_output = direct_model(direct_hidden_states).last_hidden_state
    offloaded_loss = offloaded_output.float().square().mean()
    direct_loss = direct_output.float().square().mean()

    offloaded_loss.backward()
    direct_loss.backward()

    assert torch.equal(offloaded_loss, direct_loss)
    assert hidden_states.grad is not None
    assert torch.equal(hidden_states.grad.cpu(), direct_hidden_states.grad.cpu())
    offloaded_grads = [
        parameter.grad.detach().clone()
        for parameter in offloaded_model.parameters()
        if parameter.requires_grad
    ]
    direct_grads = [
        parameter.grad.detach().cpu().clone()
        for parameter in direct_model.parameters()
        if parameter.requires_grad
    ]
    for offloaded_grad, direct_grad in zip(offloaded_grads, direct_grads, strict=True):
        assert torch.equal(offloaded_grad, direct_grad)

    offloaded_optimizer.step()
    direct_optimizer.step()

    for offloaded_param, direct_param in zip(
        offloaded_model.parameters(), direct_model.parameters(), strict=True
    ):
        assert offloaded_param.dtype == torch.bfloat16
        assert direct_param.dtype == torch.bfloat16
        assert torch.equal(offloaded_param.detach(), direct_param.detach().cpu())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_layers_use_transient_cuda_copies_and_keep_cpu_masters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _collect_offload_events(monkeypatch)
    model = SyntheticModel(
        synthetic_layer_stack(),
        RepeatedLayerOffloadConfig(
            enabled=True,
            execution_device="cuda",
        ),
    )
    hidden_states = torch.randn(2, 4, device="cuda", requires_grad=True)

    output = model(hidden_states).last_hidden_state
    loss = output.sum()
    loss.backward()

    assert output.device.type == "cuda"
    assert model.layers[0].proj.weight.device.type == "cpu"
    assert model.layers[0].proj.weight.is_pinned()
    assert model.layers[0].proj.weight.grad is not None
    assert model.layers[0].proj.weight.grad.device.type == "cpu"
    assert ("prefetch_cuda", 0) in events
    assert ("prefetch_wait", 0) in events
    assert ("free_forward", 0) in events
    assert ("flatten_grad", 0) in events


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_optimizer_updates_repeated_layers_on_cpu_with_adam() -> None:
    model = SyntheticModel(
        synthetic_layer_stack(width=8, depth=2),
        RepeatedLayerOffloadConfig(enabled=True),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    hidden_states = torch.randn(3, 8, device="cuda", requires_grad=True)

    before_step = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    loss = model(hidden_states).last_hidden_state.float().square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.device.type == "cpu"
        assert parameter.is_pinned()
        assert parameter.grad is not None
        assert parameter.grad.device.type == "cpu"

    optimizer.step()

    after_step = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_step, after_step, strict=True)
    )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offload_forces_checkpoint_recompute_and_reuses_device_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = _collect_offload_events(monkeypatch)
    offloaded_layers = synthetic_layer_stack(width=8, depth=3)
    direct_layers = copy.deepcopy(offloaded_layers)
    offloaded_model = SyntheticModel(
        offloaded_layers,
        RepeatedLayerOffloadConfig(enabled=True),
    )
    direct_model = SyntheticModel(direct_layers).to("cuda")
    hidden_states = torch.randn(2, 8, device="cuda", requires_grad=True)
    direct_hidden_states = hidden_states.detach().clone().requires_grad_(True)

    offloaded_output = offloaded_model(hidden_states).last_hidden_state
    direct_output = direct_model(direct_hidden_states).last_hidden_state
    offloaded_loss = offloaded_output.float().square().mean()
    direct_loss = direct_output.float().square().mean()

    offloaded_loss.backward()
    direct_loss.backward()

    assert torch.equal(offloaded_loss, direct_loss)
    assert hidden_states.grad is not None
    assert torch.equal(hidden_states.grad.cpu(), direct_hidden_states.grad.cpu())
    for offloaded_param, direct_param in zip(
        offloaded_model.parameters(), direct_model.parameters(), strict=True
    ):
        assert offloaded_param.grad is not None
        assert torch.equal(offloaded_param.grad, direct_param.grad.cpu())
    for layer_idx in range(3):
        assert events.count(("flat_parameters_transfer", layer_idx)) == 2
        assert events.count(("prefetch_backward", layer_idx)) == 0
        assert ("free_backward", layer_idx) in events


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_model_level_offload_materializes_repeated_layers_on_cpu() -> None:
    model = from_hf_config(
        llama_config(),
        attn_implementation="kernels-community/flash-attn3",
        layer_offload_config=RepeatedLayerOffloadConfig(enabled=True),
    )
    model.set_random_init()
    model.materialize("cuda")

    for name, parameter in model.pre_decoder.named_parameters(recurse=True):
        assert parameter.device.type == "cuda", name
    for name, parameter in model.post_decoder.named_parameters(
        recurse=True,
        remove_duplicate=False,
    ):
        assert parameter.device.type == "cuda", name

    layer_parameters = list(model.decoder_layers.parameters())
    assert layer_parameters
    assert all(parameter.device.type == "cpu" for parameter in layer_parameters)
