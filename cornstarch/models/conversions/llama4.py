from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask, create_chunked_causal_mask
from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

from cornstarch.models.conversions.llama import CausalLanguageForwardSpec
from cornstarch.models.forward_specs import LayerContext, _filtered_layer_kwargs
from cornstarch.models.language_model import CornstarchLanguageModel


class Llama4LanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for Llama4 text models."""

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        position_ids = kwargs.get("position_ids")
        past_key_values = kwargs.get("past_key_values")
        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if isinstance(past_key_values, Cache)
                else 0
            )
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": model.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "chunked_attention": create_chunked_causal_mask(**mask_kwargs),
            }

        return {
            "attention_mask": causal_mask_mapping,
            "past_key_values": past_key_values,
            "position_embeddings": model.pre_decoder["rotary_emb"](
                hidden_states, position_ids
            ),
            "position_ids": position_ids,
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        layer_type = model.config.layer_types[layer_idx]
        return {
            "attention_mask": context["attention_mask"][layer_type],
            "position_ids": context["position_ids"],
            "past_key_values": context["past_key_values"],
            "use_cache": False,
            "position_embeddings": context["position_embeddings"],
            **_filtered_layer_kwargs(kwargs),
        }


def convert_llama4_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchLanguageModel:
    """Convert a Llama4 text config into a meta-initialized language model."""
    with torch.device("meta"):
        hf_model = Llama4ForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        config,
        pre_decoder={
            "embed_tokens": hf_model.model.embed_tokens,
            "rotary_emb": hf_model.model.rotary_emb,
        },
        decoder_layers=hf_model.model.layers,
        post_decoder={"norm": hf_model.model.norm, "lm_head": hf_model.lm_head},
        hf_to_cornstarch_prefixes=(
            ("model.embed_tokens.", "pre_decoder.embed_tokens."),
            ("model.rotary_emb.", "pre_decoder.rotary_emb."),
            ("model.layers.", "decoder_layers."),
            ("model.norm.", "post_decoder.norm."),
            ("lm_head.", "post_decoder.lm_head."),
        ),
        hf_model_factory=Llama4ForCausalLM,
        forward_spec=Llama4LanguageForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
