from __future__ import annotations

import copy

import torch
from transformers import PretrainedConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from new_cornstarch.models.language_model import CornstarchLanguageModel


def convert_deepseek_v3_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a DeepSeek-V3 config into a meta-initialized language model."""
    with torch.device("meta"):
        hf_model = DeepseekV3ForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        config,
        pre_decoder={"embed_tokens": hf_model.model.embed_tokens},
        repeated_layers=hf_model.model.layers,
        post_decoder={"norm": hf_model.model.norm, "lm_head": hf_model.lm_head},
        hf_to_cornstarch_prefixes=(
            ("model.embed_tokens.", "pre_decoder.embed_tokens."),
            ("model.layers.", "repeated_layers.layers."),
            ("model.norm.", "post_decoder.norm."),
            ("lm_head.", "post_decoder.lm_head."),
        ),
        hf_model_factory=DeepseekV3ForCausalLM,
        attn_implementation=attn_implementation,
    )
