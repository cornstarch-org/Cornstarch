from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from new_cornstarch.models.conversions.llama import CausalLanguageForwardSpec
from new_cornstarch.models.forward_specs import LayerContext, _filtered_layer_kwargs
from new_cornstarch.models.language_model import CornstarchLanguageModel


class QwenMoeLanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for Qwen3.5 MoE text models."""

    output_cls = MoeCausalLMOutputWithPast

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
            position_ids = position_ids.view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rotary_position_ids = position_ids

        causal_mask = create_causal_mask(
            config=model.config,
            inputs_embeds=hidden_states,
            attention_mask=kwargs.get("attention_mask"),
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        attention_mask = kwargs.get("attention_mask")
        linear_attn_mask = attention_mask
        if (
            past_key_values is not None
            and hasattr(past_key_values, "has_previous_state")
            and past_key_values.has_previous_state()
        ) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None

        position_embeddings = model.pre_decoder["rotary_emb"](
            hidden_states, rotary_position_ids
        )
        return {
            "causal_mask": causal_mask,
            "linear_attn_mask": linear_attn_mask,
            "past_key_values": past_key_values,
            "position_embeddings": position_embeddings,
            "text_position_ids": text_position_ids,
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        layer_types = getattr(model.config, "layer_types", [])
        layer_mask = (
            context["linear_attn_mask"]
            if layer_idx < len(layer_types) and layer_types[layer_idx] == "linear_attention"
            else context["causal_mask"]
        )
        return {
            "attention_mask": layer_mask,
            "position_ids": context["text_position_ids"],
            "past_key_values": context["past_key_values"],
            "use_cache": False,
            "position_embeddings": context["position_embeddings"],
            **_filtered_layer_kwargs(kwargs),
        }

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> MoeCausalLMOutputWithPast:
        output = super().build_output(model, hidden_states, context, **kwargs)
        return MoeCausalLMOutputWithPast(
            loss=output.loss,
            aux_loss=None,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            router_logits=None,
        )


def convert_qwen3_5_moe_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
) -> CornstarchLanguageModel:
    """Convert a Qwen3.5 MoE text config into a Cornstarch language model."""
    with torch.device("meta"):
        hf_model = Qwen3_5MoeForCausalLM(copy.deepcopy(config))
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
        hf_model_factory=Qwen3_5MoeForCausalLM,
        forward_spec=QwenMoeLanguageForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
    )
