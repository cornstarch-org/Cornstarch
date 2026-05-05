from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    MoeCausalLMOutputWithPast,
    Seq2SeqModelOutput,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import load_balancing_loss_func
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures
from transformers.models.whisper.modeling_whisper import _compute_mask_indices


LayerContext = dict[str, Any]


class TransformerForwardSpec:
    """Model-family hooks for the shared Cornstarch transformer forward loop."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        return {}

    def should_skip_layer(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> bool:
        return False

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> Any:
        return BaseModelOutput(last_hidden_state=hidden_states)


def run_transformer_forward(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: TransformerForwardSpec,
    kwargs: dict[str, Any],
) -> Any:
    """Run the shared Cornstarch-owned transformer layer loop."""
    hidden_states = spec.embed_inputs(model, **kwargs)
    loop_kwargs = dict(kwargs)
    loop_kwargs.pop("hidden_states", None)
    context = spec.prepare_layer_context(model, hidden_states, **loop_kwargs)

    for layer_idx, layer in enumerate(layers):
        if spec.should_skip_layer(model, layer_idx, context, **loop_kwargs):
            continue
        layer_output = layer(
            hidden_states,
            **spec.get_layer_kwargs(model, layer_idx, context, **loop_kwargs),
        )
        hidden_states = spec.process_layer_output(
            model, layer_idx, layer_output, context, **loop_kwargs
        )

    hidden_states = spec.finalize_hidden_states(model, hidden_states, context, **loop_kwargs)
    return spec.build_output(model, hidden_states, context, **loop_kwargs)


def _validate_input_choice(input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None) -> None:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


def _filtered_layer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "is_causal",
        "max_length_q",
        "max_length_k",
        "num_items_in_batch",
        "output_attentions",
        "output_hidden_states",
        "output_router_logits",
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


class CausalLanguageForwardSpec(TransformerForwardSpec):
    """Native forward spec for Llama/Qwen/DeepSeek-style decoder-only models."""

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
        output_router_logits = kwargs.get("output_router_logits")
        if output_router_logits is None:
            output_router_logits = getattr(model.config, "output_router_logits", False)

        output = super().build_output(model, hidden_states, context, **kwargs)
        aux_loss = None
        router_logits = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                router_logits,
                getattr(model.config, "num_experts", None),
                getattr(model.config, "num_experts_per_tok", 2),
                kwargs.get("attention_mask"),
            )
            if output.loss is not None:
                output.loss = output.loss + getattr(model.config, "router_aux_loss_coef", 0.0) * aux_loss

        return MoeCausalLMOutputWithPast(
            loss=output.loss,
            aux_loss=aux_loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            router_logits=router_logits,
        )


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


@dataclass(frozen=True)
class Qwen3VLVisionForwardSpec(TransformerForwardSpec):
    """Native forward spec for Qwen3-VL vision encoders."""

    num_grid_per_side: int
    spatial_merge_size: int
    deepstack_visual_indexes: tuple[int, ...]

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        hidden_states = model.pre_encoder["patch_embed"](kwargs["hidden_states"])
        pos_embeds = self._fast_pos_embed_interpolate(model, kwargs["grid_thw"])
        return hidden_states + pos_embeds

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        grid_thw = kwargs["grid_thw"]
        rotary_pos_emb = self._rot_pos_emb(model, grid_thw)
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        return {
            "cu_seqlens": F.pad(cu_seqlens, (1, 0), value=0),
            "deepstack_features": [],
            "position_embeddings": (emb.cos(), emb.sin()),
        }

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {
            "cu_seqlens": context["cu_seqlens"],
            "position_embeddings": context["position_embeddings"],
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
        hidden_states = super().process_layer_output(
            model, layer_idx, layer_output, context, **kwargs
        )
        if layer_idx in self.deepstack_visual_indexes:
            merger_index = self.deepstack_visual_indexes.index(layer_idx)
            deepstack_feature = model.post_encoder["deepstack_merger_list"][merger_index](
                hidden_states
            )
            context["deepstack_features"].append(deepstack_feature)
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> BaseModelOutputWithDeepstackFeatures:
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=model.post_encoder["merger"](hidden_states),
            deepstack_features=context["deepstack_features"],
        )

    def _fast_pos_embed_interpolate(
        self, model: nn.Module, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = model.pre_encoder["pos_embed"].weight.device
        idx_list: list[list[int]] = [[] for _ in range(4)]
        weight_list: list[list[float]] = [[] for _ in range(4)]

        for _, height, width in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, height)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, width)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list,
            dtype=model.pre_encoder["pos_embed"].weight.dtype,
            device=device,
        )
        pos_embeds = model.pre_encoder["pos_embed"](idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        for pos_embed, frames, height, width in zip(
            patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=True
        ):
            pos_embed = pos_embed.repeat(frames, 1)
            pos_embed = (
                pos_embed.view(
                    frames,
                    height // self.spatial_merge_size,
                    self.spatial_merge_size,
                    width // self.spatial_merge_size,
                    self.spatial_merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def _rot_pos_emb(self, model: nn.Module, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(height, width) for _, height, width in grid_thw_list)
        freq_table = model.pre_encoder["rotary_pos_emb"](max_hw)
        device = freq_table.device
        total_tokens = sum(frames * height * width for frames, height, width in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h = height // self.spatial_merge_size
            merged_w = width // self.spatial_merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(self.spatial_merge_size, device=device)
            intra_col = torch.arange(self.spatial_merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * self.spatial_merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * self.spatial_merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)
            col_idx = col_idx.expand(
                merged_h, merged_w, self.spatial_merge_size, self.spatial_merge_size
            ).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        return freq_table[pos_ids].flatten(1)


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
