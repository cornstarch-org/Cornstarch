from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM

from new_cornstarch.models.conversions.llama import CausalLanguageForwardSpec
from new_cornstarch.models.forward_specs import LayerContext, _filtered_layer_kwargs
from new_cornstarch.models.language_model import CornstarchLanguageModel


class GlmMoeDsaLanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for GLM MoE DSA decoder-only models."""

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

        causal_mask = create_causal_mask(
            config=model.config,
            inputs_embeds=hidden_states,
            attention_mask=kwargs.get("attention_mask"),
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = model.pre_decoder["rotary_emb"](
            hidden_states, position_ids=position_ids
        )
        return {
            "attention_mask": causal_mask,
            "past_key_values": past_key_values,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "topk_indices": None,
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "attention_mask": context["attention_mask"],
            "position_ids": context["position_ids"],
            "past_key_values": context["past_key_values"],
            "use_cache": False,
            "position_embeddings": context["position_embeddings"],
            "prev_topk_indices": context["topk_indices"],
            **_filtered_layer_kwargs(kwargs),
        }

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states, topk_indices = layer_output
        context["topk_indices"] = topk_indices
        return hidden_states


def convert_glm_moe_dsa_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a GLM MoE DSA config into a meta-initialized language model."""
    with torch.device("meta"):
        hf_model = GlmMoeDsaForCausalLM(copy.deepcopy(config))
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
        hf_model_factory=GlmMoeDsaForCausalLM,
        forward_spec=GlmMoeDsaLanguageForwardSpec(),
        attn_implementation=attn_implementation,
    )
