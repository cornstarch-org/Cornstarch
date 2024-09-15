# Copied from https://github.com/hpcaitech/ColossalAI/blob/v0.4.2/colossalai/shardformer/modeling/whisper.py

from typing import List, Optional, Tuple, Union

import torch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention
from colossalai.shardformer.shard import ShardConfig
from torch import nn
from transformers.cache_utils import EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperEncoder,
    WhisperSdpaAttention,
    logger,
)


class WhisperPipelineForwards:
    @staticmethod
    def whisper_encoder_forward(
        self: WhisperEncoder,
        input_features: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[torch.FloatTensor] = None,
        all_attentions: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: Optional[ShardConfig] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Process inputs if at the first stage of encoder.
        if stage_manager.is_first_stage():
            inputs_embeds = nn.functional.gelu(self.conv1(input_features))
            inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

            inputs_embeds = inputs_embeds.permute(0, 2, 1)
            embed_pos = self.embed_positions.weight

            hidden_states = inputs_embeds + embed_pos
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # check if head_mask has a correct number of layers specified if desired
            if head_mask is not None:
                assert head_mask.size()[0] == (
                    len(self.layers)
                ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        else:
            if hidden_states is None:
                raise ValueError(
                    "hidden_states shouldn't be None for stages other than the first stage of encoder/decoder."
                )

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, encoder_layer in enumerate(
            self.layers[start_idx:end_idx], start=start_idx
        ):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if stage_manager.is_last_stage():
            hidden_states = self.layer_norm(hidden_states)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, encoder_states, all_attentions]
                    if v is not None
                )
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )

        else:
            return {"hidden_states": hidden_states, "head_mask": head_mask}


class WhisperForwards:
    def whisper_flash_attention_forward(
        self: Union[WhisperAttention, WhisperSdpaAttention],
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[dict] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "The `static` cache implementation is not compatible with `attn_implementation='flash_attention_2'`. "
                "Use `attn_implementation='sdpa'` in the meantime, and open an issue at https://github.com/huggingface/transformers"
            )
        # WhisperFlashAttention2 attention does not support output_attentions
        if output_attentions:
            raise ValueError(
                "WhisperFlashAttention2 attention does not support output_attentions"
            )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        # use key_value_states if cross attention
        current_states = (
            key_value_states if key_value_states is not None else hidden_states
        )
        if is_cross_attention and past_key_value and is_updated:
            # reuse k,v, cross_attentions
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape(self.k_proj(current_states), -1, bsz)
            value_states = self._shape(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )

        # For encoder, attention_mask is None
        if attention_mask is None:
            attention_mask = {}
        attn_output = ColoAttention.attention(
            query_states,
            key_states,
            value_states,
            **attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
