from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig


def llama_config() -> PretrainedConfig:
    """Load the tiny Llama config used by model wrapper tests."""
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")


def qwen3_5_config() -> PretrainedConfig:
    """Load the tiny Qwen3.5 text config used by model wrapper tests."""
    return AutoConfig.from_pretrained(
        "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"
    ).text_config


def qwen3_5_moe_config() -> PretrainedConfig:
    """Build the compact synthetic Qwen3.5 MoE config used in tests."""
    return Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "linear_attention"],
        tie_word_embeddings=False,
    )


def deepseek_v3_config() -> PretrainedConfig:
    """Load the tiny DeepSeek-V3 config used by model wrapper tests."""
    return AutoConfig.from_pretrained("trl-internal-testing/tiny-DeepseekV3ForCausalLM")


def clip_vision_config() -> PretrainedConfig:
    """Load the tiny CLIP vision config used by model wrapper tests."""
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-CLIPModel").vision_config


def siglip2_vision_config() -> PretrainedConfig:
    """Build a reduced SigLIP2 vision config for lightweight tests."""
    config = AutoConfig.from_pretrained("google/siglip2-base-patch16-naflex").vision_config
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    return config


def qwen3_vl_vision_config() -> PretrainedConfig:
    """Load the tiny Qwen3-VL vision config used by model wrapper tests."""
    return AutoConfig.from_pretrained("tiny-random/qwen3-vl").vision_config


def whisper_config() -> PretrainedConfig:
    """Load the tiny Whisper config used by model wrapper tests."""
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-WhisperModel")
