from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
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


def gemma4_config() -> PretrainedConfig:
    """Build the compact synthetic Gemma4 text config used in tests."""
    return Gemma4TextConfig(
        vocab_size=32,
        vocab_size_per_layer_input=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        tie_word_embeddings=False,
    )


def glm_moe_dsa_config() -> PretrainedConfig:
    """Build the compact synthetic GLM MoE DSA config used in tests."""
    return GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=4,
        q_lora_rank=4,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_topk=2,
        index_head_dim=8,
        index_n_heads=2,
        mlp_layer_types=["dense", "sparse"],
        indexer_types=["full", "full"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
    )


def llama4_config() -> PretrainedConfig:
    """Build the compact synthetic Llama4 text config used in tests."""
    return Llama4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        intermediate_size_mlp=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_layers=[1],
        no_rope_layers=[1, 0],
        layer_types=["chunked_attention", "full_attention"],
        attention_chunk_size=4,
        tie_word_embeddings=False,
    )


def nemotron_h_config() -> PretrainedConfig:
    """Build the compact synthetic Nemotron-H config used in tests."""
    return NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "moe", "mlp"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        mlp_hidden_act="silu",
        use_mamba_kernels=False,
        ssm_state_size=4,
        mamba_num_heads=4,
        mamba_head_dim=4,
        n_groups=2,
        conv_kernel=3,
        chunk_size=4,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        rescale_prenorm_residual=False,
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


def gemma4_vision_config() -> PretrainedConfig:
    """Build a reduced Gemma4 vision config for lightweight tests."""
    return Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        patch_size=2,
        pooling_kernel_size=1,
        position_embedding_size=8,
        standardize=False,
    )


def whisper_config() -> PretrainedConfig:
    """Load the tiny Whisper config used by model wrapper tests."""
    return AutoConfig.from_pretrained("hf-internal-testing/tiny-random-WhisperModel")


def gemma4_audio_config() -> PretrainedConfig:
    """Build a reduced Gemma4 audio config for lightweight tests."""
    return Gemma4AudioConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        subsampling_conv_channels=(8, 4),
        attention_chunk_size=2,
        attention_context_left=3,
        attention_context_right=0,
        conv_kernel_size=3,
        output_proj_dims=12,
    )
