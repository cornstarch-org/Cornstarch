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
        hf_model,
        config,
        repeated_layers=hf_model.model.layers,
        attn_implementation=attn_implementation,
    )
