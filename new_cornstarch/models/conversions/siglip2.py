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
        config,
        pre_encoder={"embeddings": hf_model.embeddings},
        repeated_layers=hf_model.encoder.layers,
        post_encoder={
            "post_layernorm": hf_model.post_layernorm,
            "head": hf_model.head,
        },
        hf_to_cornstarch_prefixes=(
            ("embeddings.", "pre_encoder.embeddings."),
            ("encoder.layers.", "repeated_layers.layers."),
            ("post_layernorm.", "post_encoder.post_layernorm."),
            ("head.", "post_encoder.head."),
        ),
        hf_model_factory=Siglip2VisionModel,
        attn_implementation=attn_implementation,
    )
