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
    """Convert a CLIP vision config into a meta-initialized Cornstarch encoder."""
    with torch.device("meta"):
        hf_model = CLIPVisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        config,
        pre_encoder={
            "embeddings": hf_model.embeddings,
            "pre_layrnorm": hf_model.pre_layrnorm,
        },
        encoder_layers=hf_model.encoder.layers,
        post_encoder={"post_layernorm": hf_model.post_layernorm},
        hf_to_cornstarch_prefixes=(
            ("embeddings.", "pre_encoder.embeddings."),
            ("pre_layrnorm.", "pre_encoder.pre_layrnorm."),
            ("encoder.layers.", "encoder_layers."),
            ("post_layernorm.", "post_encoder.post_layernorm."),
        ),
        hf_model_factory=CLIPVisionModel,
        forward_impl=hf_model.forward,
        attn_implementation=attn_implementation,
    )
