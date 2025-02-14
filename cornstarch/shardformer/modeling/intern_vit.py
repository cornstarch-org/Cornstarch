from typing import Optional, Tuple, Union

import torch
from colossalai.shardformer.shard.shard_config import ShardConfig
from torch import nn
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)

from cornstarch.models.intern_vit.modeling_intern_vit import (
    InternAttention,
    InternVisionModel,
)


class InternVisionModelForwards:
    @staticmethod
    def intern_vit_model_forward(
        self: InternVisionModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_embeds: Optional[torch.FloatTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        shard_config: Optional[ShardConfig] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None and pixel_embeds is None:
                raise ValueError("You have to specify pixel_values or pixel_embeds")

            if pixel_embeds is not None:
                hidden_states = pixel_embeds
            else:
                if len(pixel_values.shape) == 4:
                    hidden_states = self.embeddings(pixel_values)
                else:
                    raise ValueError(f"wrong pixel_values size: {pixel_values.shape}")

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.encoder.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.encoder.layers))

        for encoder_layer in self.encoder.layers[start_idx:end_idx]:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.encoder.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer, hidden_states, use_reentrant=False
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {"hidden_states": hidden_states}
            if output_hidden_states:
                outputs["encoder_states"] = encoder_states
            return outputs

        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class InternVisionAttentionForwards:
    @staticmethod
    def flash_attention_forward(
        self: InternAttention,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.embed_dim]
        value_states = qkv[..., query_pos + self.embed_dim :]

        # [batch_size, tgt_len, embed_dim] -> [batch_size, tgt_len, num_heads, head_dim]
        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            query_length=tgt_len,
            is_causal=False,
            dropout=self.attn_drop.p,
            softmax_scale=self.scale,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output

    @staticmethod
    def eager_attention_forward(
        self: InternAttention,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        def shape(self: InternAttention, tensor: torch.Tensor, seq_len: int, bsz: int):
            return (
                tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        bsz, tgt_len, _ = hidden_states.size()

        qkv = self.qkv(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.embed_dim]
        value_states = qkv[..., query_pos + self.embed_dim :]

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = shape(self, query_states, tgt_len, bsz).view(*proj_shape)
        key_states = shape(self, key_states, tgt_len, bsz).view(*proj_shape)
        value_states = shape(self, value_states, tgt_len, bsz).view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states * self.scale, key_states.transpose(1, 2))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.proj(attn_output)
        attn_output = self.proj_drop(attn_output)

        return attn_output
