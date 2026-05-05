from __future__ import annotations

import copy

import torch
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

from new_cornstarch.models.vision_encoder import CornstarchVisionEncoder


def convert_siglip2_vision_config(
    config: Siglip2VisionConfig,
    attn_implementation: str | None = None,
) -> CornstarchVisionEncoder:
    """Convert a SigLIP2 vision config into a Cornstarch vision encoder."""
    with torch.device("meta"):
        hf_model = Siglip2VisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        hf_model,
        config,
        repeated_layers=hf_model.encoder.layers,
        attn_implementation=attn_implementation,
    )
