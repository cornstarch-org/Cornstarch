from __future__ import annotations

import copy

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from new_cornstarch.models.vision_encoder import CornstarchVisionEncoder


def convert_qwen3_vl_vision_config(
    config: Qwen3VLVisionConfig,
    attn_implementation: str | None = None,
) -> CornstarchVisionEncoder:
    """Convert a Qwen3-VL vision config into a Cornstarch vision encoder."""
    with torch.device("meta"):
        hf_model = Qwen3VLVisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        hf_model,
        config,
        repeated_layers=hf_model.blocks,
        attn_implementation=attn_implementation,
    )
