from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from cornstarch.models.conversions.llama import CausalLanguageForwardSpec
from cornstarch.models.forward_specs import LayerContext, _filtered_layer_kwargs
from cornstarch.models.language_model import CornstarchLanguageModel


class DeepseekV4LanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for DeepSeek-V4 decoder-only models.

    DeepSeek-V4 keeps the standard decoder-only checkpoint layout, but its
    transformer loop runs on parallel hyper-connection streams and collapses them
    through ``hc_head`` before the final RMS norm. It also uses the V4
    sliding-window mask and cache object because compressed attention layers
    expect V4-specific cache state even when callers do not request returned
    generation caches.
    """

    output_cls = MoeCausalLMOutputWithPast

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        inputs_embeds = super().embed_inputs(model, **kwargs)
        return inputs_embeds.unsqueeze(2).expand(
            -1, -1, model.config.hc_mult, -1
        ).contiguous()

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        position_ids = kwargs.get("position_ids")
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None:
            past_key_values = DynamicCache(config=model.config)
        return_cache = past_key_values if kwargs.get("use_cache") else None

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
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=model.config,
                inputs_embeds=hidden_states[..., 0, :],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        position_embeddings = model.pre_decoder["rotary_emb"](
            hidden_states[..., 0, :],
            position_ids=position_ids,
            layer_type="main",
        )
        return {
            "attention_mask": causal_mask,
            "past_key_values": past_key_values,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "return_cache": return_cache,
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
            "input_ids": kwargs.get("input_ids"),
            **_filtered_layer_kwargs(kwargs),
        }

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        hidden_states = model.post_decoder["hc_head"](hidden_states)
        return model.post_decoder["norm"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> MoeCausalLMOutputWithPast:
        output = super().build_output(model, hidden_states, context, **kwargs)
        return self.output_cls(
            loss=output.loss,
            aux_loss=None,
            logits=output.logits,
            past_key_values=context.get("return_cache"),
            hidden_states=None,
            attentions=None,
            router_logits=None,
        )


def convert_deepseek_v4_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchLanguageModel:
    """Convert a DeepSeek-V4 config into a meta-initialized language model."""
    with torch.device("meta"):
        hf_model = DeepseekV4ForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        config,
        pre_decoder={
            "embed_tokens": hf_model.model.embed_tokens,
            "rotary_emb": hf_model.model.rotary_emb,
        },
        decoder_layers=hf_model.model.layers,
        post_decoder={
            "hc_head": hf_model.model.hc_head,
            "norm": hf_model.model.norm,
            "lm_head": hf_model.lm_head,
        },
        hf_to_cornstarch_prefixes=(
            ("model.embed_tokens.", "pre_decoder.embed_tokens."),
            ("model.rotary_emb.", "pre_decoder.rotary_emb."),
            ("model.layers.", "decoder_layers."),
            ("model.hc_head.", "post_decoder.hc_head."),
            ("model.norm.", "post_decoder.norm."),
            ("lm_head.", "post_decoder.lm_head."),
        ),
        hf_model_factory=DeepseekV4ForCausalLM,
        forward_spec=DeepseekV4LanguageForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
