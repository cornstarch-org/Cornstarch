from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from new_cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)
from new_cornstarch.models.language_model import CornstarchLanguageModel


def _validate_input_choice(input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> None:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


class CausalLanguageForwardSpec(TransformerForwardSpec):
    """Native decoder-only forward behavior shared by causal language families.

    Llama, Qwen, DeepSeek, and related models all follow the same Cornstarch
    structure: token embeddings and rotary helpers live in ``pre_decoder``,
    transformer blocks live in ``decoder_layers``, and normalization plus the
    LM head live in ``post_decoder``. This spec supplies the family behavior that
    the shared Cornstarch loop needs to run that structure without borrowing a
    bound Hugging Face root-model ``forward`` method.

    The spec builds input embeddings, causal masks, position ids, and rotary
    position embeddings before layer execution; forwards only the generic layer
    kwargs supported by the reused Hugging Face leaf blocks; applies the final
    decoder norm; and returns a ``CausalLMOutputWithPast`` with logits and
    optional training loss. Optional capture outputs remain intentionally unset
    unless Cornstarch adds explicit support for them.
    """

    output_cls = CausalLMOutputWithPast

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        _validate_input_choice(input_ids, inputs_embeds)
        if inputs_embeds is None:
            inputs_embeds = model.pre_decoder["embed_tokens"](input_ids)
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
        position_embeddings = model.pre_decoder["rotary_emb"](
            hidden_states, position_ids=position_ids
        )
        return {
            "attention_mask": causal_mask,
            "past_key_values": past_key_values,
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
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
            **_filtered_layer_kwargs(kwargs),
        }

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return model.post_decoder["norm"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        logits_to_keep = kwargs.get("logits_to_keep", 0)
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = model.post_decoder["lm_head"](hidden_states[:, slice_indices, :])
        labels = kwargs.get("labels")
        loss = None
        if labels is not None:
            loss_kwargs = dict(kwargs)
            loss_kwargs.pop("labels", None)
            loss_kwargs.pop("vocab_size", None)
            loss = ForCausalLMLoss(
                logits=logits,
                labels=labels,
                vocab_size=model.config.vocab_size,
                **loss_kwargs,
            )
        return self.output_cls(
            loss=loss,
            logits=logits,
            past_key_values=context.get("past_key_values"),
            hidden_states=None,
            attentions=None,
        )


def convert_llama_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
) -> CornstarchLanguageModel:
    """Convert a Llama config into a meta-initialized Cornstarch language model."""
    with torch.device("meta"):
        hf_model = LlamaForCausalLM(copy.deepcopy(config))
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
        hf_model_factory=LlamaForCausalLM,
        forward_spec=CausalLanguageForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
    )
