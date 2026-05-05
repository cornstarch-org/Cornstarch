from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionModel

from new_cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)
from new_cornstarch.models.vision_encoder import CornstarchVisionEncoder


class ClipVisionForwardSpec(TransformerForwardSpec):
    """Native forward spec for CLIP vision encoders."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        hidden_states = model.pre_encoder["embeddings"](
            kwargs.get("pixel_values"),
            interpolate_pos_encoding=kwargs.get("interpolate_pos_encoding", False),
        )
        return model.pre_encoder["pre_layrnorm"](hidden_states)

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {"attention_mask": kwargs.get("attention_mask"), **_filtered_layer_kwargs(kwargs)}

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        pooled_output = hidden_states[:, 0, :]
        context["pooler_output"] = model.post_encoder["post_layernorm"](pooled_output)
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> BaseModelOutputWithPooling:
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=context["pooler_output"],
        )


def convert_clip_vision_config(
    config: CLIPVisionConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
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
        forward_spec=ClipVisionForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
