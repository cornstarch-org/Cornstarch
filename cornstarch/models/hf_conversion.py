from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

import torch
from transformers import AutoConfig, PretrainedConfig

from cornstarch.models.conversions.clip import convert_clip_vision_config
from cornstarch.models.conversions.deepseek_v3 import convert_deepseek_v3_config
from cornstarch.models.conversions.deepseek_v4 import convert_deepseek_v4_config
from cornstarch.models.conversions.gemma4 import (
    convert_gemma4_audio_config,
    convert_gemma4_config,
    convert_gemma4_vision_config,
)
from cornstarch.models.conversions.glm_moe_dsa import convert_glm_moe_dsa_config
from cornstarch.models.conversions.llama import convert_llama_config
from cornstarch.models.conversions.llama4 import convert_llama4_config
from cornstarch.models.conversions.nemotron_h import convert_nemotron_h_config
from cornstarch.models.conversions.qwen3_5 import convert_qwen3_5_config
from cornstarch.models.conversions.qwen3_5_moe import convert_qwen3_5_moe_config
from cornstarch.models.conversions.qwen3_vl import convert_qwen3_vl_vision_config
from cornstarch.models.conversions.siglip2 import convert_siglip2_vision_config
from cornstarch.models.conversions.whisper import convert_whisper_config
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from cornstarch.models.layer_compile import RepeatedLayerCompileConfig
from cornstarch.models.layer_offload import RepeatedLayerOffloadConfig
from cornstarch.models.model_base import CornstarchModelBase


ModelKind = Literal["language", "vision", "audio"]
Converter = Callable[..., CornstarchModelBase]


@dataclass(frozen=True)
class _ConversionSpec:
    """Bind one HF config identity to its converter and unified root kind.

    Model-family knowledge belongs at this conversion boundary. A converter may
    understand Hugging Face's concrete leaf layout, but its output is always one
    of Cornstarch's unified roots. Lifecycle and parallelization code therefore
    never dispatches on the original model family.
    """

    kind: ModelKind
    converter: Converter


_CONVERSION_BY_MODEL_TYPE: dict[str, _ConversionSpec] = {
    "clip_vision_model": _ConversionSpec("vision", convert_clip_vision_config),
    "deepseek_v3": _ConversionSpec("language", convert_deepseek_v3_config),
    "deepseek_v4": _ConversionSpec("language", convert_deepseek_v4_config),
    "gemma4_audio": _ConversionSpec("audio", convert_gemma4_audio_config),
    "gemma4_text": _ConversionSpec("language", convert_gemma4_config),
    "gemma4_vision": _ConversionSpec("vision", convert_gemma4_vision_config),
    "glm_moe_dsa": _ConversionSpec("language", convert_glm_moe_dsa_config),
    "llama": _ConversionSpec("language", convert_llama_config),
    "llama4_text": _ConversionSpec("language", convert_llama4_config),
    "nemotron_h": _ConversionSpec("language", convert_nemotron_h_config),
    "qwen3_5_moe_text": _ConversionSpec("language", convert_qwen3_5_moe_config),
    "qwen3_5_text": _ConversionSpec("language", convert_qwen3_5_config),
    "qwen3_vl_vision": _ConversionSpec("vision", convert_qwen3_vl_vision_config),
    "siglip2_vision_model": _ConversionSpec("vision", convert_siglip2_vision_config),
    "whisper": _ConversionSpec("audio", convert_whisper_config),
}

# A few nested HF configs use a parent model_type that does not identify the
# leaf encoder. Exact config types disambiguate those cases without teaching the
# rest of Cornstarch about Hugging Face family names.
_CONVERSION_BY_CONFIG_TYPE: tuple[tuple[type[PretrainedConfig], _ConversionSpec], ...] = (
    (Siglip2VisionConfig, _ConversionSpec("vision", convert_siglip2_vision_config)),
    (Qwen3VLVisionConfig, _ConversionSpec("vision", convert_qwen3_vl_vision_config)),
)


def _conversion_spec(config: PretrainedConfig) -> _ConversionSpec:
    for config_type, spec in _CONVERSION_BY_CONFIG_TYPE:
        if isinstance(config, config_type):
            return spec
    model_type = getattr(config, "model_type", None)
    try:
        return _CONVERSION_BY_MODEL_TYPE[model_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model architecture for Cornstarch conversion: {model_type}"
        ) from error


def from_hf_config(
    config: PretrainedConfig,
    model_kind: ModelKind | None = None,
    attn_implementation: str | None = None,
    layer_offload_config: RepeatedLayerOffloadConfig | None = None,
    layer_compile_config: RepeatedLayerCompileConfig | None = None,
) -> CornstarchModelBase:
    """Create the matching Cornstarch model for a Hugging Face config.

    The family converter only translates Hugging Face's leaf layout. Every
    language family returns ``CornstarchLanguageModel`` and every encoder family
    returns the unified Cornstarch encoder representation, all still on
    ``meta``. ``model_kind`` is an optional assertion, not a fallback converter:
    unsupported configs must provide an explicit converter rather than being
    silently interpreted as Llama, CLIP, or Whisper.
    """
    spec = _conversion_spec(config)
    if model_kind is not None and model_kind != spec.kind:
        raise ValueError(
            f"Config {type(config).__name__} converts to a {spec.kind} model, "
            f"not the requested {model_kind} model."
        )
    return spec.converter(
        config,
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )


def from_pretrained_config(
    model_name_or_path: str | Path,
    model_kind: ModelKind | None = None,
    attn_implementation: str | None = None,
    layer_offload_config: RepeatedLayerOffloadConfig | None = None,
    layer_compile_config: RepeatedLayerCompileConfig | None = None,
    trust_remote_code: bool = False,
    **kwargs,
) -> CornstarchModelBase:
    """Load a Hugging Face config and convert it into a Cornstarch model."""
    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
    )
    return from_hf_config(
        config,
        model_kind=model_kind,
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )


def load_hf_state_dict(
    model: CornstarchModelBase,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load Hugging Face-format weights into a Cornstarch model."""
    return model.load_hf_state_dict(state_dict, strict=strict)


def to_hf_state_dict(model: CornstarchModelBase) -> dict[str, torch.Tensor]:
    """Export a materialized Cornstarch model as a Hugging Face state dict."""
    return model.to_hf_state_dict()


def infer_model_kind(config: PretrainedConfig) -> ModelKind:
    """Infer whether a supported Hugging Face config is language, vision, or audio."""
    return _conversion_spec(config).kind
