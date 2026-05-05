from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig

from new_cornstarch.models import from_hf_config


ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"


def _llama_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")


def _qwen3_5_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration").text_config


def _deepseek_v3_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("trl-internal-testing/tiny-DeepseekV3ForCausalLM")


def _clip_vision_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-CLIPModel").vision_config


def _siglip2_vision_config() -> PretrainedConfig:
    config = AutoConfig.from_pretrained("google/siglip2-base-patch16-naflex").vision_config
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    return config


def _qwen3_vl_vision_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("tiny-random/qwen3-vl").vision_config


def _whisper_config() -> PretrainedConfig:
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-WhisperModel")


MODEL_CONFIG_FACTORIES: list[tuple[str, Callable[[], PretrainedConfig]]] = [
    ("llama", _llama_config),
    ("qwen3_5", _qwen3_5_config),
    ("deepseek_v3", _deepseek_v3_config),
    ("clip_vision", _clip_vision_config),
    ("siglip2_vision", _siglip2_vision_config),
    ("qwen3_vl_vision", _qwen3_vl_vision_config),
    ("whisper", _whisper_config),
]


def _model_config_params() -> list[object]:
    return [
        pytest.param(config_factory, id=model_id)
        for model_id, config_factory in MODEL_CONFIG_FACTORIES
    ]


def _repeated_layer_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    return list(model.repeated_layers.parameters()) + list(model.repeated_layers.buffers())


def _assert_repeated_layers_meta(model: torch.nn.Module) -> None:
    tensors = _repeated_layer_tensors(model)
    assert tensors
    assert all(tensor.is_meta for tensor in tensors)


def _assert_repeated_layers_on_device(
    model: torch.nn.Module, device: torch.device
) -> None:
    tensors = _repeated_layer_tensors(model)
    assert tensors
    assert all(not tensor.is_meta for tensor in tensors)
    assert all(tensor.device.type == device.type for tensor in tensors)


@pytest.mark.parametrize("config_factory", _model_config_params())
def test_repeated_layers_are_meta_when_model_is_initialized(
    config_factory: Callable[[], PretrainedConfig],
) -> None:
    cornstarch_model = from_hf_config(
        config_factory(), attn_implementation=ATTN_IMPLEMENTATION
    )

    _assert_repeated_layers_meta(cornstarch_model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for layer materialization")
@pytest.mark.parametrize("config_factory", _model_config_params())
def test_materialize_layers_stores_repeated_layers_on_cuda(
    config_factory: Callable[[], PretrainedConfig],
) -> None:
    cornstarch_model = from_hf_config(
        config_factory(), attn_implementation=ATTN_IMPLEMENTATION
    )
    _assert_repeated_layers_meta(cornstarch_model)

    cuda_device = torch.device("cuda")
    cornstarch_model.materialize_layers(cuda_device)

    _assert_repeated_layers_on_device(cornstarch_model, cuda_device)
