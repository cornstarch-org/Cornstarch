from typing import Optional, Tuple, Union

import torch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import (
    SiglipAttention,
    SiglipVisionModel,
    SiglipVisionTransformer,
)


class SiglipVisionPipelineForwards:
    @staticmethod
    def siglip_vision_transformer_forward(
        self: SiglipVisionTransformer,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[list[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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

        if stage_manager.is_first_stage():
            hidden_states = self.embeddings(
                pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
            )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        start_idx, end_idx = stage_index
        for encoder_layer in self.encoder.layers[start_idx:end_idx]:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.encoder.gradient_checkpointing and self.training:
                layer_outputs = self.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,  # attention_mask
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if stage_manager.is_last_stage():
            encoder_outputs = BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooler_output = self.head(last_hidden_state) if self.use_head else None
            if not return_dict:
                return (last_hidden_state, pooler_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        # always return dict for intermediate stage
        return {"hidden_states": hidden_states}

    @staticmethod
    def siglip_vision_model_forward(
        self: SiglipVisionModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[list[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return SiglipVisionPipelineForwards.siglip_vision_transformer_forward(
            self.vision_model,
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict,
            interpolate_pos_encoding,
            stage_manager,
            hidden_states,
            stage_index,
            shard_config,
        )


class SiglipVisionForwards:
    @staticmethod
    def clip_flash_attention_forward(
        self: SiglipAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.size()

        # [batch_size, q_len, embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [batch_size, q_len, embed_dim] -> [batch_size, q_len, num_heads, head_dim]
        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_output = ColoAttention.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            scale=self.scale,
        )

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None
