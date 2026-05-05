from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from transformers import PretrainedConfig

from new_cornstarch.models import from_hf_config
from new_tests.model.model_configs import (
    clip_vision_config,
    deepseek_v3_config,
    llama_config,
    qwen3_5_config,
    qwen3_vl_vision_config,
    siglip2_vision_config,
    whisper_config,
)


ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"


MODEL_CONFIG_FACTORIES: list[tuple[str, Callable[[], PretrainedConfig]]] = [
    ("llama", llama_config),
    ("qwen3_5", qwen3_5_config),
    ("deepseek_v3", deepseek_v3_config),
    ("clip_vision", clip_vision_config),
    ("siglip2_vision", siglip2_vision_config),
    ("qwen3_vl_vision", qwen3_vl_vision_config),
    ("whisper", whisper_config),
]


def _model_config_params() -> list[object]:
    """Create pytest parameters with readable IDs for each supported config."""
    return [
        pytest.param(config_factory, id=model_id)
        for model_id, config_factory in MODEL_CONFIG_FACTORIES
    ]


def _materialize_device_params() -> list[object]:
    """Create pytest parameters for devices that can materialize layer tensors."""
    return [
        pytest.param(torch.device("cpu"), id="cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is required for layer materialization",
            ),
            id="cuda",
        ),
    ]


def _model_layer_stack(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Return the architecture-specific transformer layer stack."""
    if hasattr(model, "decoder_layers"):
        return model.decoder_layers
    return model.encoder_layers


def _model_layer_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    """Collect parameters and buffers from a model's transformer layer stack."""
    layer_stack = _model_layer_stack(model)
    return list(layer_stack.parameters()) + list(layer_stack.buffers())


def _assert_model_layers_meta(model: torch.nn.Module) -> None:
    """Assert every transformer-layer tensor is still on the meta device."""
    tensors = _model_layer_tensors(model)
    assert tensors
    assert all(tensor.is_meta for tensor in tensors)


@pytest.mark.parametrize("config_factory", _model_config_params())
def test_model_layers_are_meta_when_model_is_initialized(
    config_factory: Callable[[], PretrainedConfig],
) -> None:
    """Verify converted models leave transformer layers meta-initialized."""
    cornstarch_model = from_hf_config(
        config_factory(), attn_implementation=ATTN_IMPLEMENTATION
    )

    assert isinstance(_model_layer_stack(cornstarch_model), torch.nn.ModuleList)
    _assert_model_layers_meta(cornstarch_model)


@pytest.mark.parametrize("config_factory", _model_config_params())
@pytest.mark.parametrize("device", _materialize_device_params())
def test_materialize_layers_stores_model_layers_on_device(
    config_factory: Callable[[], PretrainedConfig],
    device: torch.device,
) -> None:
    """Verify layer-only materialization works with plain ModuleList layers."""
    cornstarch_model = from_hf_config(
        config_factory(), attn_implementation=ATTN_IMPLEMENTATION
    )
    _assert_model_layers_meta(cornstarch_model)

    cornstarch_model.materialize_layers(device)

    tensors = _model_layer_tensors(cornstarch_model)
    assert tensors
    assert all(not tensor.is_meta for tensor in tensors)
    assert all(tensor.device.type == device.type for tensor in tensors)


@pytest.mark.parametrize("materialized_device", _materialize_device_params())
def test_offload_layers_to_cpu_skips_unmaterialized_layers(
    materialized_device: torch.device,
) -> None:
    """Verify layer offload preserves still-meta decoder layers."""
    cornstarch_model = from_hf_config(
        llama_config(), attn_implementation=ATTN_IMPLEMENTATION
    )
    assert len(cornstarch_model.decoder_layers) > 1

    materialized_layer = cornstarch_model.decoder_layers[0]
    materialized_layer.to_empty(device=materialized_device)

    cornstarch_model.offload_layers_to_cpu([0])

    materialized_tensors = list(materialized_layer.parameters()) + list(
        materialized_layer.buffers()
    )
    meta_tensors = (
        list(cornstarch_model.decoder_layers[1].parameters())
        + list(cornstarch_model.decoder_layers[1].buffers())
    )
    assert materialized_tensors
    assert meta_tensors
    assert all(
        not tensor.is_meta and tensor.device.type == "cpu" for tensor in materialized_tensors
    )
    assert all(tensor.is_meta for tensor in meta_tensors)
