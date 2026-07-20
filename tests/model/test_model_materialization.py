from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from cornstarch.models import from_hf_config
from tests.model.model_configs import (
    clip_vision_config,
    deepseek_v3_config,
    deepseek_v4_config,
    gemma4_audio_config,
    gemma4_config,
    gemma4_vision_config,
    glm_moe_dsa_config,
    llama4_config,
    llama_config,
    nemotron_h_config,
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
    ("deepseek_v4", deepseek_v4_config),
    ("gemma4", gemma4_config),
    ("glm_moe_dsa", glm_moe_dsa_config),
    ("llama4", llama4_config),
    ("nemotron_h", nemotron_h_config),
    ("clip_vision", clip_vision_config),
    ("siglip2_vision", siglip2_vision_config),
    ("qwen3_vl_vision", qwen3_vl_vision_config),
    ("gemma4_vision", gemma4_vision_config),
    ("whisper", whisper_config),
    ("gemma4_audio", gemma4_audio_config),
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


def _reference_hf_state_dict() -> dict[str, torch.Tensor]:
    """Materialize a tiny Llama and return its Hugging Face-keyed weights."""
    reference = from_hf_config(llama_config(), attn_implementation=ATTN_IMPLEMENTATION)
    reference.set_random_init()
    reference.materialize("cpu")
    return {key: tensor.detach().clone() for key, tensor in reference.to_hf_state_dict().items()}


def test_set_checkpoint_init_from_state_dict_round_trips() -> None:
    """A staged HF state dict materializes into matching Cornstarch weights."""
    hf_state_dict = _reference_hf_state_dict()

    model = from_hf_config(llama_config(), attn_implementation=ATTN_IMPLEMENTATION)
    model.set_checkpoint_init(state_dict=hf_state_dict)
    model.materialize("cpu")

    materialized = model.to_hf_state_dict()
    assert set(materialized) == set(hf_state_dict)
    for key, expected in hf_state_dict.items():
        torch.testing.assert_close(materialized[key], expected)


def test_set_checkpoint_init_from_hub_id_round_trips(monkeypatch) -> None:
    """``model_name_or_path`` downloads + merges safetensors, then materializes.

    The real Hub download is monkeypatched so ``pytest tests`` stays offline; the
    code path under test (HF-to-Cornstarch mapping + device/dtype move + load) is
    identical to a genuine Hub materialization.
    """
    hf_state_dict = _reference_hf_state_dict()

    def _fake_download(model_name_or_path, source_to_local):
        assert model_name_or_path == "fake/tiny-llama"
        return {
            local_key: hf_state_dict[source_key].clone()
            for source_key, local_key in source_to_local.items()
        }

    monkeypatch.setattr(
        "cornstarch.models.model_base.CornstarchModelBase._download_and_load_safetensors",
        staticmethod(_fake_download),
    )

    model = from_hf_config(llama_config(), attn_implementation=ATTN_IMPLEMENTATION)
    model.set_checkpoint_init(model_name_or_path="fake/tiny-llama")
    model.materialize("cpu")

    materialized = model.to_hf_state_dict()
    assert set(materialized) == set(hf_state_dict)
    for key, expected in hf_state_dict.items():
        torch.testing.assert_close(materialized[key], expected)


def test_pipeline_local_checkpoint_keys_use_global_layer_indices() -> None:
    """A PP-local layer zero loads the corresponding global checkpoint layer."""
    hf_state_dict = _reference_hf_state_dict()
    model = from_hf_config(llama_config(), attn_implementation=ATTN_IMPLEMENTATION)
    model.set_checkpoint_init(state_dict=hf_state_dict)

    # Mirror apply_pipeline_parallel without starting a process group: this
    # stage owns original layer one, exposed locally as decoder_layers.0.
    model.decoder_layers = nn.ModuleList([model.decoder_layers[1]])
    model._pipeline_layer_offset = 1
    model.materialize("cpu")

    actual = model.state_dict()["decoder_layers.0.self_attn.q_proj.weight"]
    expected = hf_state_dict["model.layers.1.self_attn.q_proj.weight"]
    torch.testing.assert_close(actual, expected)


def test_materialize_handles_duplicate_tied_parameter_names() -> None:
    """Verify tied Llama embeddings do not leave the LM head on meta."""
    cornstarch_model = from_hf_config(
        llama_config(), attn_implementation=ATTN_IMPLEMENTATION
    )
    cornstarch_model.set_random_init()

    cornstarch_model.materialize("cpu")

    meta_parameters = [
        name
        for name, parameter in cornstarch_model.named_parameters(remove_duplicate=False)
        if parameter.is_meta
    ]
    assert meta_parameters == []


def test_materialize_handles_mixed_meta_and_materialized_tensors() -> None:
    """A concrete buffer must not hide parameters that still need allocation."""
    model = from_hf_config(llama_config(), attn_implementation=ATTN_IMPLEMENTATION)
    model.register_buffer("materialized_sentinel", torch.ones(1), persistent=False)
    model.set_random_init()

    model.materialize("cpu")

    tensors = list(model.parameters()) + list(model.buffers())
    assert tensors
    assert all(not tensor.is_meta for tensor in tensors)
