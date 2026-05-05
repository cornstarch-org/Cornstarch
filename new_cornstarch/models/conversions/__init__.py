"""Conversion helpers for supported Hugging Face model families."""

from new_cornstarch.models.conversions.clip import convert_clip_vision_config
from new_cornstarch.models.conversions.deepseek_v3 import convert_deepseek_v3_config
from new_cornstarch.models.conversions.gemma4 import (
    convert_gemma4_audio_config,
    convert_gemma4_config,
    convert_gemma4_vision_config,
)
from new_cornstarch.models.conversions.glm_moe_dsa import convert_glm_moe_dsa_config
from new_cornstarch.models.conversions.llama import convert_llama_config
from new_cornstarch.models.conversions.llama4 import convert_llama4_config
from new_cornstarch.models.conversions.nemotron_h import convert_nemotron_h_config
from new_cornstarch.models.conversions.qwen3_5 import convert_qwen3_5_config
from new_cornstarch.models.conversions.qwen3_5_moe import convert_qwen3_5_moe_config
from new_cornstarch.models.conversions.qwen3_vl import convert_qwen3_vl_vision_config
from new_cornstarch.models.conversions.siglip2 import convert_siglip2_vision_config
from new_cornstarch.models.conversions.whisper import convert_whisper_config

__all__ = [
    "convert_clip_vision_config",
    "convert_deepseek_v3_config",
    "convert_gemma4_audio_config",
    "convert_gemma4_config",
    "convert_gemma4_vision_config",
    "convert_glm_moe_dsa_config",
    "convert_llama_config",
    "convert_llama4_config",
    "convert_nemotron_h_config",
    "convert_qwen3_5_config",
    "convert_qwen3_5_moe_config",
    "convert_qwen3_vl_vision_config",
    "convert_siglip2_vision_config",
    "convert_whisper_config",
]
