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
        config,
        pre_encoder={
            "patch_embed": hf_model.patch_embed,
            "pos_embed": hf_model.pos_embed,
            "rotary_pos_emb": hf_model.rotary_pos_emb,
        },
        repeated_layers=hf_model.blocks,
        post_encoder={
            "merger": hf_model.merger,
            "deepstack_merger_list": hf_model.deepstack_merger_list,
        },
        hf_to_cornstarch_prefixes=(
            ("patch_embed.", "pre_encoder.patch_embed."),
            ("pos_embed.", "pre_encoder.pos_embed."),
            ("rotary_pos_emb.", "pre_encoder.rotary_pos_emb."),
            ("blocks.", "repeated_layers."),
            ("merger.", "post_encoder.merger."),
            ("deepstack_merger_list.", "post_encoder.deepstack_merger_list."),
        ),
        hf_model_factory=Qwen3VLVisionModel,
        forward_impl=hf_model.forward,
        attn_implementation=attn_implementation,
    )
