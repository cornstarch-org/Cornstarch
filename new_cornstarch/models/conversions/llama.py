from __future__ import annotations

import copy

import torch
from transformers import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from new_cornstarch.models.language_model import CornstarchLanguageModel


def convert_llama_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a Llama config into a meta-initialized Cornstarch language model."""
    with torch.device("meta"):
        hf_model = LlamaForCausalLM(copy.deepcopy(config))
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
        hf_model_factory=LlamaForCausalLM,
        attn_implementation=attn_implementation,
    )
