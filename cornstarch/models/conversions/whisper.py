from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperModel, _compute_mask_indices

from cornstarch.models.encoder_base import CornstarchEncoder
from cornstarch.models.forward_specs import (
    LayerContext,
    TransformerForwardSpec,
    _filtered_layer_kwargs,
)


class WhisperSeq2SeqForwardSpec(TransformerForwardSpec):
    """Native encoder loop plus HF decoder leaf for WhisperModel compatibility."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("encoder_outputs") is not None:
            raise NotImplementedError("encoder_outputs passthrough is not implemented for Cornstarch Whisper.")

        input_features = self._mask_input_features(
            model,
            kwargs.get("input_features"),
            attention_mask=kwargs.get("attention_mask"),
        )
        expected_seq_length = model.config.max_source_positions * model.pre_encoder["conv1"].stride[0] * model.pre_encoder["conv2"].stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[-1]}. Make sure to pad the input mel features "
                f"to {expected_seq_length}."
            )

        inputs_embeds = F.gelu(model.pre_encoder["conv1"](input_features))
        inputs_embeds = F.gelu(model.pre_encoder["conv2"](inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(
            model.pre_encoder["embed_positions"].num_embeddings,
            device=inputs_embeds.device,
        )
        hidden_states = inputs_embeds + model.pre_encoder["embed_positions"](all_positions)
        return F.dropout(hidden_states, p=getattr(model.config, "dropout", 0.0), training=model.training)

    def should_skip_layer(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> bool:
        layerdrop = getattr(model.config, "encoder_layerdrop", 0.0)
        return bool(model.training and layerdrop > 0 and torch.rand([]) < layerdrop)

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {"attention_mask": None, **_filtered_layer_kwargs(kwargs)}

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return model.post_encoder["encoder_layer_norm"](hidden_states)

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> Seq2SeqModelOutput:
        encoder_outputs = BaseModelOutput(last_hidden_state=hidden_states)
        decoder_outputs = model.post_encoder["decoder"](
            input_ids=kwargs.get("decoder_input_ids"),
            attention_mask=kwargs.get("decoder_attention_mask"),
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            past_key_values=kwargs.get("past_key_values"),
            inputs_embeds=kwargs.get("decoder_inputs_embeds"),
            position_ids=kwargs.get("decoder_position_ids"),
            use_cache=kwargs.get("use_cache"),
            **_filtered_layer_kwargs(kwargs),
        )
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def _mask_input_features(
        self,
        model: nn.Module,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not getattr(model.config, "apply_spec_augment", True):
            return input_features

        batch_size, hidden_size, sequence_length = input_features.size()
        if model.config.mask_time_prob > 0 and model.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=model.config.mask_time_prob,
                mask_length=model.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=model.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(
                mask_time_indices, device=input_features.device, dtype=torch.bool
            )
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features = input_features.masked_fill(mask_time_indices, 0)

        if model.config.mask_feature_prob > 0 and model.training:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=model.config.mask_feature_prob,
                mask_length=model.config.mask_feature_length,
                min_masks=model.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(
                mask_feature_indices, device=input_features.device, dtype=torch.bool
            )
            input_features = input_features.masked_fill(mask_feature_indices, 0)

        return input_features


def convert_whisper_config(
    config: WhisperConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchEncoder:
    """Convert a Whisper config into a meta-initialized Cornstarch audio encoder."""
    with torch.device("meta"):
        hf_model = WhisperModel(copy.deepcopy(config))
    return CornstarchEncoder(
        config,
        pre_encoder={
            "conv1": hf_model.encoder.conv1,
            "conv2": hf_model.encoder.conv2,
            "embed_positions": hf_model.encoder.embed_positions,
        },
        encoder_layers=hf_model.encoder.layers,
        post_encoder={
            "encoder_layer_norm": hf_model.encoder.layer_norm,
            "decoder": hf_model.decoder,
        },
        hf_to_cornstarch_prefixes=(
            ("encoder.conv1.", "pre_encoder.conv1."),
            ("encoder.conv2.", "pre_encoder.conv2."),
            ("encoder.embed_positions.", "pre_encoder.embed_positions."),
            ("encoder.layers.", "encoder_layers."),
            ("encoder.layer_norm.", "post_encoder.encoder_layer_norm."),
            ("decoder.", "post_encoder.decoder."),
        ),
        hf_model_factory=WhisperModel,
        forward_spec=WhisperSeq2SeqForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
