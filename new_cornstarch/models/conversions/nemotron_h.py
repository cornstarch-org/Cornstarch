from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

from new_cornstarch.models.conversions.llama import (
    CausalLanguageForwardSpec,
    _validate_input_choice,
)
from new_cornstarch.models.forward_specs import LayerContext, _filtered_layer_kwargs
from new_cornstarch.models.language_model import CornstarchLanguageModel


class NemotronHLanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for Nemotron-H hybrid Mamba/attention blocks."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        _validate_input_choice(input_ids, inputs_embeds)
        if inputs_embeds is None:
            inputs_embeds = model.pre_decoder["embeddings"](input_ids)
        return inputs_embeds

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
        return {
            "attention_mask": {
                "attention": causal_mask,
                "mamba": self._update_mamba_mask(
                    kwargs.get("attention_mask"), past_key_values
                ),
                "mlp": None,
                "moe": None,
            },
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        block_type = model.decoder_layers[layer_idx].block_type
        return {
            "attention_mask": context["attention_mask"][block_type],
            "position_ids": context["position_ids"],
            "past_key_values": context["past_key_values"],
            "use_cache": False,
            **_filtered_layer_kwargs(kwargs),
        }

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return model.post_decoder["norm_f"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        output = super().build_output(model, hidden_states, context, **kwargs)
        return CausalLMOutputWithPast(
            loss=output.loss,
            logits=output.logits.float(),
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _update_mamba_mask(
        self,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor | None:
        if (
            past_key_values is not None
            and hasattr(past_key_values, "has_previous_state")
            and past_key_values.has_previous_state()
        ) or (attention_mask is not None and torch.all(attention_mask == 1)):
            return None
        return attention_mask


def convert_nemotron_h_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a Nemotron-H config into a meta-initialized hybrid language model."""
    with torch.device("meta"):
        hf_model = NemotronHForCausalLM(copy.deepcopy(config))
    return CornstarchLanguageModel(
        config,
        pre_decoder={"embeddings": hf_model.model.embeddings},
        decoder_layers=hf_model.model.layers,
        post_decoder={"norm_f": hf_model.model.norm_f, "lm_head": hf_model.lm_head},
        hf_to_cornstarch_prefixes=(
            ("model.embeddings.", "pre_decoder.embeddings."),
            ("model.layers.", "decoder_layers."),
            ("model.norm_f.", "post_decoder.norm_f."),
            ("lm_head.", "post_decoder.lm_head."),
        ),
        hf_model_factory=NemotronHForCausalLM,
        forward_spec=NemotronHLanguageForwardSpec(),
        attn_implementation=attn_implementation,
    )
