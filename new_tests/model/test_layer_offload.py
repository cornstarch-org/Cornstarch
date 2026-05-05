from __future__ import annotations

import copy

import pytest
import torch

from new_cornstarch.models import RepeatedLayerOffloadConfig, from_hf_config
from new_tests.model.model_configs import llama_config
from new_tests.model.synthetic_model_configs import (
    SyntheticModel,
    synthetic_layer_stack,
)


def _event_index(events: list[tuple[str, int]], event: str, layer_idx: int) -> int:
    return events.index((event, layer_idx))


def test_layer_offload_config_threads_through_model_factory() -> None:
    config = RepeatedLayerOffloadConfig(enabled=True)

    model = from_hf_config(
        llama_config(),
        attn_implementation="kernels-community/flash-attn3",
        layer_offload_config=config,
    )

    assert model.layer_offload_config is config
    assert model.uses_layer_offload


def test_repeated_layer_offload_is_disabled_by_default() -> None:
    model = from_hf_config(
        llama_config(), attn_implementation="kernels-community/flash-attn3"
    )

    assert model.layer_offload_config is None
    assert not model.uses_layer_offload


def test_enabled_layer_offload_rejects_cpu_execution_device() -> None:
    with pytest.raises(ValueError, match="CUDA execution device"):
        RepeatedLayerOffloadConfig(enabled=True, execution_device="cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_forward_prefetches_next_layer_and_records_cpu_activations() -> None:
    events: list[tuple[str, int]] = []
    model = SyntheticModel(
        synthetic_layer_stack(),
        RepeatedLayerOffloadConfig(
            enabled=True,
            event_callback=lambda event, layer_idx: events.append((event, layer_idx)),
        ),
    )

    output = model(torch.randn(2, 4, device="cuda", requires_grad=True))

    assert output.last_hidden_state.device.type == "cuda"
    assert [event for event in events if event[0] == "activation_cpu"] == [
        ("activation_cpu", 0),
        ("activation_cpu", 1),
        ("activation_cpu", 2),
    ]
    assert _event_index(events, "prefetch_forward", 1) < _event_index(
        events, "run_forward", 0
    )
    assert _event_index(events, "prefetch_forward", 2) < _event_index(
        events, "run_forward", 1
    )
    assert ("free_forward", 0) in events
    assert ("free_forward", 1) in events
    assert ("free_forward", 2) in events


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Repeated layer offload executes on CUDA."
)
def test_offloaded_backward_prefetches_previous_layer_and_accumulates_cpu_grads() -> None:
    events: list[tuple[str, int]] = []
    offloaded_layers = synthetic_layer_stack()
    direct_layers = copy.deepcopy(offloaded_layers)
    offloaded_model = SyntheticModel(
        offloaded_layers,
        RepeatedLayerOffloadConfig(
            enabled=True,
            event_callback=lambda event, layer_idx: events.append((event, layer_idx)),
        ),
    )
    direct_model = SyntheticModel(direct_layers).to("cuda")
    hidden_states = torch.randn(2, 4, device="cuda", requires_grad=True)
    direct_hidden_states = hidden_states.detach().clone().requires_grad_(True)

    offloaded_loss = offloaded_model(hidden_states).last_hidden_state.sum()
    direct_loss = direct_model(direct_hidden_states).last_hidden_state.sum()

    offloaded_loss.backward()
    direct_loss.backward()

    assert _event_index(events, "prefetch_backward", 1) < _event_index(
        events, "run_backward", 2
    )
    assert _event_index(events, "prefetch_backward", 0) < _event_index(
        events, "run_backward", 1
    )
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
def test_offloaded_layers_use_transient_cuda_copies_and_keep_cpu_masters() -> None:
    model = SyntheticModel(
        synthetic_layer_stack(),
        RepeatedLayerOffloadConfig(enabled=True, execution_device="cuda"),
    )
    hidden_states = torch.randn(2, 4, device="cuda", requires_grad=True)

    loss = model(hidden_states).last_hidden_state.sum()
    loss.backward()

    assert model.layers[0].proj.weight.device.type == "cpu"
    assert model.layers[0].proj.weight.grad is not None
    assert model.layers[0].proj.weight.grad.device.type == "cpu"
