from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

import new_cornstarch.models.layer_compile as layer_compile_module
from new_cornstarch.models import (
    RepeatedLayerCompileConfig,
    RepeatedLayerOffloadConfig,
    from_hf_config,
)
from new_tests.model.model_configs import llama_config
from new_tests.model.synthetic_model_configs import SyntheticModel, synthetic_layer_stack


def _run_forward_backward(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]]:
    model.zero_grad(set_to_none=True)
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    output = model(hidden_states).last_hidden_state
    output.square().sum().backward()
    parameter_grads = [
        None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for parameter in model.parameters()
    ]
    assert hidden_states.grad is not None
    return (
        output.detach().cpu(),
        hidden_states.grad.detach().cpu().clone(),
        parameter_grads,
    )


def _assert_run_results_close(
    actual: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]],
    expected: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor | None]],
) -> None:
    actual_output, actual_input_grad, actual_parameter_grads = actual
    expected_output, expected_input_grad, expected_parameter_grads = expected
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_input_grad, expected_input_grad)
    assert len(actual_parameter_grads) == len(expected_parameter_grads)
    for actual_grad, expected_grad in zip(
        actual_parameter_grads,
        expected_parameter_grads,
        strict=True,
    ):
        if actual_grad is None or expected_grad is None:
            assert actual_grad is expected_grad
        else:
            torch.testing.assert_close(actual_grad, expected_grad)


def test_repeated_layer_compile_is_enabled_by_default() -> None:
    model = from_hf_config(
        llama_config(),
        attn_implementation="kernels-community/flash-attn3",
    )

    assert model.layer_compile_config.enabled


def test_repeated_layer_compile_config_can_disable_compile() -> None:
    config = RepeatedLayerCompileConfig(enabled=False)

    model = from_hf_config(
        llama_config(),
        attn_implementation="kernels-community/flash-attn3",
        layer_compile_config=config,
    )

    assert model.layer_compile_config is config
    assert not model.layer_compile_config.enabled


def test_direct_compiled_layers_match_eager_outputs_and_gradients() -> None:
    eager_model = SyntheticModel(synthetic_layer_stack()).eval()
    compiled_model = SyntheticModel(
        synthetic_layer_stack(),
        layer_compile_config=RepeatedLayerCompileConfig(backend="eager"),
    ).eval()
    compiled_model.load_state_dict(eager_model.state_dict())
    hidden_states = torch.randn(2, 4)

    eager_result = _run_forward_backward(eager_model, hidden_states)
    compiled_result = _run_forward_backward(compiled_model, hidden_states)

    _assert_run_results_close(compiled_result, eager_result)


def test_direct_compile_caches_one_callable_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, Any]] = []

    def fake_compile(fn: Callable[..., Any], **kwargs: Any) -> Callable[..., Any]:
        compile_calls.append(kwargs)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(layer_compile_module.torch, "compile", fake_compile)
    model = SyntheticModel(
        synthetic_layer_stack(depth=2),
        layer_compile_config=RepeatedLayerCompileConfig(backend="eager"),
    ).eval()
    hidden_states = torch.randn(2, 4)

    model(hidden_states)
    model(hidden_states)

    assert len(compile_calls) == 2
    assert [call["backend"] for call in compile_calls] == ["eager", "eager"]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Layer offload executes on CUDA."
)
def test_offloaded_compiled_layers_match_eager_outputs_and_gradients() -> None:
    eager_model = SyntheticModel(
        synthetic_layer_stack(),
        layer_offload_config=RepeatedLayerOffloadConfig(enabled=True),
    ).eval()
    compiled_model = SyntheticModel(
        synthetic_layer_stack(),
        layer_offload_config=RepeatedLayerOffloadConfig(enabled=True),
        layer_compile_config=RepeatedLayerCompileConfig(backend="eager"),
    ).eval()
    compiled_model.load_state_dict(eager_model.state_dict())
    hidden_states = torch.randn(2, 4, device="cuda")

    eager_result = _run_forward_backward(eager_model, hidden_states)
    compiled_result = _run_forward_backward(compiled_model, hidden_states)

    _assert_run_results_close(compiled_result, eager_result)
