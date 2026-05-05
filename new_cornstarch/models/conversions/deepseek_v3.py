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
    with torch.device("meta"):
        hf_model = DeepseekV3ForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        hf_model,
        config,
        repeated_layers=hf_model.model.layers,
        attn_implementation=attn_implementation,
    )
