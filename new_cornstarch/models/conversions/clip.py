from __future__ import annotations

import copy

import torch
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionModel

from new_cornstarch.models.vision_encoder import CornstarchVisionEncoder


def convert_clip_vision_config(
    config: CLIPVisionConfig,
    attn_implementation: str | None = None,
) -> CornstarchVisionEncoder:
    with torch.device("meta"):
        hf_model = CLIPVisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        hf_model,
        config,
        repeated_layers=hf_model.encoder.layers,
        attn_implementation=attn_implementation,
    )
