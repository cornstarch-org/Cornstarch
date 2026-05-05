from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioModel,
    Gemma4AudioModelOutput,
    Gemma4ForCausalLM,
    Gemma4VisionModel,
    sliding_window_mask_function,
)

from new_cornstarch.models.audio_encoder import CornstarchAudioEncoder
from new_cornstarch.models.conversions.llama import (
    CausalLanguageForwardSpec,
    _validate_input_choice,
)
from new_cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)
from new_cornstarch.models.language_model import CornstarchLanguageModel
from new_cornstarch.models.vision_encoder import CornstarchVisionEncoder


class _Gemma4AudioMaskConverter(nn.Module):
    """Convert Gemma4 audio 4D masks into blocked local-attention masks."""

    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config

    def forward(self, mask_4d: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = mask_4d.shape
        device = mask_4d.device
        chunk_size = self.config.attention_chunk_size
        max_past_horizon = self.config.attention_context_left - 1
        max_future_horizon = self.config.attention_context_right
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        mask_4d = F.pad(mask_4d, (0, pad_amount, 0, pad_amount), value=False)
        mask_5d = mask_4d.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        mask_5d = F.pad(mask_5d, (max_past_horizon, max_future_horizon), value=False)

        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(
            chunk_size + max_past_horizon + max_future_horizon, device=device
        )
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(
            batch_size, 1, -1, chunk_size, -1
        )
        return mask_5d.gather(-1, kv_indices)


class Gemma4LanguageForwardSpec(CausalLanguageForwardSpec):
    """Native forward spec for Gemma4 text models."""

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
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        position_embeddings = {
            layer_type: model.pre_decoder["rotary_emb"](
                hidden_states, position_ids, layer_type
            )
            for layer_type in model.pre_decoder["rotary_emb"].layer_types
        }
        return {
            "attention_mask": causal_mask_mapping,
            "past_key_values": past_key_values,
            "per_layer_inputs": self._prepare_per_layer_inputs(
                model,
                input_ids=kwargs.get("input_ids"),
                inputs_embeds=hidden_states,
                per_layer_inputs=kwargs.get("per_layer_inputs"),
            ),
            "position_embeddings": position_embeddings,
            "position_ids": position_ids,
            "shared_kv_states": {},
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        layer_type = model.config.layer_types[layer_idx]
        per_layer_inputs = context["per_layer_inputs"]
        per_layer_input = (
            per_layer_inputs[:, :, layer_idx, :]
            if per_layer_inputs is not None
            else None
        )
        return {
            "attention_mask": context["attention_mask"][layer_type],
            "past_key_values": context["past_key_values"],
            "per_layer_input": per_layer_input,
            "position_embeddings": context["position_embeddings"][layer_type],
            "position_ids": context["position_ids"],
            "shared_kv_states": context["shared_kv_states"],
            **_filtered_layer_kwargs(kwargs),
        }

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> CausalLMOutputWithPast:
        output = super().build_output(model, hidden_states, context, **kwargs)
        if model.config.final_logit_softcapping is None:
            return output
        logits = output.logits / model.config.final_logit_softcapping
        logits = torch.tanh(logits) * model.config.final_logit_softcapping
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
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
        )

    def _prepare_per_layer_inputs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not getattr(model.config, "hidden_size_per_layer_input", 0):
            return None
        if per_layer_inputs is None:
            if input_ids is None:
                raise RuntimeError(
                    "Gemma4 per-layer embeddings require input_ids when inputs_embeds "
                    "are provided to Cornstarch."
                )
            per_layer_inputs = model.pre_decoder["embed_tokens_per_layer"](input_ids).reshape(
                *input_ids.shape,
                model.config.num_hidden_layers,
                model.config.hidden_size_per_layer_input,
            )
        per_layer_projection = (
            model.pre_decoder["per_layer_model_projection"](inputs_embeds)
            * model.config.hidden_size**-0.5
        )
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            model.config.num_hidden_layers,
            model.config.hidden_size_per_layer_input,
        )
        per_layer_projection = model.pre_decoder["per_layer_projection_norm"](
            per_layer_projection
        )
        return (per_layer_projection + per_layer_inputs) * 2.0**-0.5


class Gemma4VisionForwardSpec(TransformerForwardSpec):
    """Native forward spec for Gemma4 vision encoders."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        pixel_position_ids = kwargs["pixel_position_ids"]
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        return model.pre_encoder["patch_embedder"](
            kwargs["pixel_values"], pixel_position_ids, padding_positions
        )

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        pixel_position_ids = kwargs["pixel_position_ids"]
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        attention_mask = create_bidirectional_mask(
            config=model.config,
            inputs_embeds=hidden_states,
            attention_mask=~padding_positions,
        )
        return {
            "attention_mask": attention_mask,
            "padding_positions": padding_positions,
            "position_embeddings": model.pre_encoder["rotary_emb"](
                hidden_states, pixel_position_ids
            ),
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "attention_mask": context["attention_mask"],
            "position_embeddings": context["position_embeddings"],
            "position_ids": kwargs["pixel_position_ids"],
            **_filtered_layer_kwargs(kwargs),
        }

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> BaseModelOutputWithPast:
        pooling_kernel_size = model.config.pooling_kernel_size
        output_length = kwargs["pixel_values"].shape[-2] // (
            pooling_kernel_size * pooling_kernel_size
        )
        hidden_states, pooler_mask = model.post_encoder["pooler"](
            hidden_states=hidden_states,
            pixel_position_ids=kwargs["pixel_position_ids"],
            padding_positions=context["padding_positions"],
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class Gemma4AudioForwardSpec(TransformerForwardSpec):
    """Native forward spec for Gemma4 audio encoders."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        hidden_states, output_mask = model.pre_encoder["subsample_conv_projection"](
            kwargs["input_features"], kwargs.get("attention_mask")
        )
        self._output_mask = output_mask
        return hidden_states

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        output_mask = self._output_mask
        attention_mask = create_bidirectional_mask(
            config=model.config,
            inputs_embeds=hidden_states,
            attention_mask=output_mask,
            and_mask_function=sliding_window_mask_function(
                (
                    model.config.attention_context_left - 1,
                    model.config.attention_context_right,
                )
            ),
        )
        attention_mask = model.pre_encoder["mask_converter"](attention_mask)
        return {
            "attention_mask": attention_mask,
            "output_mask": output_mask,
            "position_embeddings": model.pre_encoder["rel_pos_enc"](hidden_states),
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "attention_mask": context["attention_mask"],
            "position_embeddings": context["position_embeddings"],
            **_filtered_layer_kwargs(kwargs),
        }

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return model.post_encoder["output_proj"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> Gemma4AudioModelOutput:
        return Gemma4AudioModelOutput(
            last_hidden_state=hidden_states,
            attention_mask=context["output_mask"],
        )


def convert_gemma4_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
) -> CornstarchLanguageModel:
    """Convert a Gemma4 text config into a meta-initialized language model."""
    with torch.device("meta"):
        hf_model = Gemma4ForCausalLM(copy.deepcopy(config))

    pre_decoder = {
        "embed_tokens": hf_model.model.embed_tokens,
        "rotary_emb": hf_model.model.rotary_emb,
    }
    prefixes = [
        ("model.embed_tokens.", "pre_decoder.embed_tokens."),
        ("model.rotary_emb.", "pre_decoder.rotary_emb."),
        ("model.layers.", "decoder_layers."),
        ("model.norm.", "post_decoder.norm."),
        ("lm_head.", "post_decoder.lm_head."),
    ]
    if getattr(config, "hidden_size_per_layer_input", 0):
        pre_decoder.update(
            {
                "embed_tokens_per_layer": hf_model.model.embed_tokens_per_layer,
                "per_layer_model_projection": hf_model.model.per_layer_model_projection,
                "per_layer_projection_norm": hf_model.model.per_layer_projection_norm,
            }
        )
        prefixes.extend(
            [
                (
                    "model.embed_tokens_per_layer.",
                    "pre_decoder.embed_tokens_per_layer.",
                ),
                (
                    "model.per_layer_model_projection.",
                    "pre_decoder.per_layer_model_projection.",
                ),
                (
                    "model.per_layer_projection_norm.",
                    "pre_decoder.per_layer_projection_norm.",
                ),
            ]
        )

    return CornstarchLanguageModel(
        config,
        pre_decoder=pre_decoder,
        decoder_layers=hf_model.model.layers,
        post_decoder={"norm": hf_model.model.norm, "lm_head": hf_model.lm_head},
        hf_to_cornstarch_prefixes=tuple(prefixes),
        hf_model_factory=Gemma4ForCausalLM,
        forward_spec=Gemma4LanguageForwardSpec(),
        attn_implementation=attn_implementation,
    )


def convert_gemma4_vision_config(
    config: Gemma4VisionConfig,
    attn_implementation: str | None = None,
) -> CornstarchVisionEncoder:
    """Convert a Gemma4 vision config into a Cornstarch vision encoder."""
    with torch.device("meta"):
        hf_model = Gemma4VisionModel(copy.deepcopy(config))
    return CornstarchVisionEncoder(
        config,
        pre_encoder={
            "patch_embedder": hf_model.patch_embedder,
            "rotary_emb": hf_model.encoder.rotary_emb,
        },
        encoder_layers=hf_model.encoder.layers,
        post_encoder={"pooler": hf_model.pooler},
        hf_to_cornstarch_prefixes=(
            ("patch_embedder.", "pre_encoder.patch_embedder."),
            ("encoder.rotary_emb.", "pre_encoder.rotary_emb."),
            ("encoder.layers.", "encoder_layers."),
            ("pooler.", "post_encoder.pooler."),
        ),
        hf_model_factory=Gemma4VisionModel,
        forward_spec=Gemma4VisionForwardSpec(),
        attn_implementation=attn_implementation,
    )


def convert_gemma4_audio_config(
    config: Gemma4AudioConfig,
    attn_implementation: str | None = None,
) -> CornstarchAudioEncoder:
    """Convert a Gemma4 audio config into a Cornstarch audio encoder."""
    with torch.device("meta"):
        hf_model = Gemma4AudioModel(copy.deepcopy(config))
    return CornstarchAudioEncoder(
        config,
        pre_encoder={
            "subsample_conv_projection": hf_model.subsample_conv_projection,
            "rel_pos_enc": hf_model.rel_pos_enc,
            "mask_converter": _Gemma4AudioMaskConverter(config),
        },
        encoder_layers=hf_model.layers,
        post_encoder={"output_proj": hf_model.output_proj},
        hf_to_cornstarch_prefixes=(
            (
                "subsample_conv_projection.",
                "pre_encoder.subsample_conv_projection.",
            ),
            ("rel_pos_enc.", "pre_encoder.rel_pos_enc."),
            ("layers.", "encoder_layers."),
            ("output_proj.", "post_encoder.output_proj."),
        ),
        hf_model_factory=Gemma4AudioModel,
        forward_spec=Gemma4AudioForwardSpec(),
        attn_implementation=attn_implementation,
    )
