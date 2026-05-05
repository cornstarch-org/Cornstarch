from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from transformers import AutoConfig, PretrainedConfig

from new_cornstarch.models.conversions.clip import convert_clip_vision_config
from new_cornstarch.models.conversions.deepseek_v3 import convert_deepseek_v3_config
from new_cornstarch.models.conversions.llama import convert_llama_config
from new_cornstarch.models.conversions.qwen3_5 import convert_qwen3_5_config
from new_cornstarch.models.conversions.qwen3_5_moe import convert_qwen3_5_moe_config
from new_cornstarch.models.conversions.qwen3_vl import convert_qwen3_vl_vision_config
from new_cornstarch.models.conversions.siglip2 import convert_siglip2_vision_config
from new_cornstarch.models.conversions.whisper import convert_whisper_config
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from new_cornstarch.models.model_base import CornstarchModelBase


def from_hf_config(
    config: PretrainedConfig,
    model_kind: str | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
) -> CornstarchModelBase:
    """Create the matching Cornstarch model for a Hugging Face config.

    The conversion builds a Cornstarch-owned module structure on the meta
    device while preserving Hugging Face checkpoint import/export semantics.
    """
    del trust_remote_code
    model_type = getattr(config, "model_type", None)
    if model_kind == "vision" or model_type == "clip_vision_model":
        return convert_clip_vision_config(config, attn_implementation=attn_implementation)
    if isinstance(config, Siglip2VisionConfig) or model_type == "siglip2_vision_model":
        return convert_siglip2_vision_config(config, attn_implementation=attn_implementation)
    if isinstance(config, Qwen3VLVisionConfig) or model_type == "qwen3_vl_vision":
        return convert_qwen3_vl_vision_config(config, attn_implementation=attn_implementation)
    if model_kind == "audio" or model_type == "whisper":
        return convert_whisper_config(config, attn_implementation=attn_implementation)
    if model_kind == "language" or model_type == "llama":
        return convert_llama_config(config, attn_implementation=attn_implementation)
    if model_type == "qwen3_5_text":
        return convert_qwen3_5_config(config, attn_implementation=attn_implementation)
    if model_type == "qwen3_5_moe_text":
        return convert_qwen3_5_moe_config(config, attn_implementation=attn_implementation)
    if model_type == "deepseek_v3":
        return convert_deepseek_v3_config(config, attn_implementation=attn_implementation)
    raise ValueError(
        f"Unsupported model architecture for Cornstarch conversion: {model_type}"
    )


def from_pretrained_config(
    model_name_or_path: str | Path,
    model_kind: str | None = None,
    attn_implementation: str | None = None,
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
        trust_remote_code=trust_remote_code,
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


def infer_model_kind(config: PretrainedConfig) -> str:
    """Infer whether a supported Hugging Face config is language, vision, or audio."""
    model_type = getattr(config, "model_type", None)
    if model_type in {"clip_vision_model", "siglip2_vision_model", "qwen3_vl_vision"}:
        return "vision"
    if model_type == "whisper":
        return "audio"
    if model_type in {"llama", "qwen3_5_text", "qwen3_5_moe_text", "deepseek_v3"}:
        return "language"
    raise ValueError(
        f"Unsupported model architecture for Cornstarch conversion: {model_type}"
    )
