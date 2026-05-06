from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel

from cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)
from cornstarch.models.vision_encoder import CornstarchVisionEncoder


class Siglip2VisionForwardSpec(TransformerForwardSpec):
    """Native forward spec for SigLIP2 NaFlex vision encoders."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        return model.pre_encoder["embeddings"](
            kwargs["pixel_values"],
            kwargs["spatial_shapes"],
        )

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        return {
            "attention_mask": create_bidirectional_mask(
                config=model.config,
                inputs_embeds=hidden_states,
                attention_mask=kwargs.get("pixel_attention_mask"),
            )
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {"attention_mask": context["attention_mask"], **_filtered_layer_kwargs(kwargs)}

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return model.post_encoder["post_layernorm"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> BaseModelOutputWithPooling:
        head = model.post_encoder["head"] if "head" in model.post_encoder else None
        pooler_output = head(hidden_states, kwargs["pixel_attention_mask"]) if head is not None else None
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
        )


def convert_siglip2_vision_config(
    config: Siglip2VisionConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchVisionEncoder:
    """Convert a SigLIP2 vision config into a Cornstarch vision encoder."""
    with torch.device("meta"):
        hf_model = Siglip2VisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        config,
        pre_encoder={"embeddings": hf_model.embeddings},
        encoder_layers=hf_model.encoder.layers,
        post_encoder={
            "post_layernorm": hf_model.post_layernorm,
            "head": hf_model.head,
        },
        hf_to_cornstarch_prefixes=(
            ("embeddings.", "pre_encoder.embeddings."),
            ("encoder.layers.", "encoder_layers."),
            ("post_layernorm.", "post_encoder.post_layernorm."),
            ("head.", "post_encoder.head."),
        ),
        hf_model_factory=Siglip2VisionModel,
        forward_spec=Siglip2VisionForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
