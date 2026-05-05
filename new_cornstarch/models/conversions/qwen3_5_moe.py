from __future__ import annotations

import copy

import torch
from transformers import PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from new_cornstarch.models.language_model import CornstarchLanguageModel


def convert_qwen3_5_moe_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a Qwen3.5 MoE text config into a Cornstarch language model."""
    with torch.device("meta"):
        hf_model = Qwen3_5MoeForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        hf_model,
        config,
        repeated_layers=hf_model.model.layers,
        attn_implementation=attn_implementation,
    )
