from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.models.clip.modeling_clip import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CLIPVisionModel,
    CLIPVisionTransformer,
)


class CLIPVisionPipelineForwards:
    """
    This class servers as a micro library for forward function substitution of CLIPVision models
    under pipeline setting.
    """

    @staticmethod
    def clip_vision_transformer_forward(
        self: CLIPVisionTransformer,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        # retrive pixel_values
        if stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            hidden_states = self.embeddings(pixel_values)
            hidden_states = self.pre_layrnorm(hidden_states)
        else:
            if hidden_states is None:
                raise ValueError(
                    "hidden_states shouldn't be None for stages other than the first stage."
                )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, encoder_layer in enumerate(
            self.encoder.layers[start_idx:end_idx], start=start_idx
        ):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.encoder.gradient_checkpointing and self.training:
                layer_outputs = self.encoder._gradient_checkpoining_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None,  # attention_mask
                    None,  # causal_attention_mask
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    causal_attention_mask=None,
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
            pooled_output = last_hidden_state[:, 0, :]
            pooled_output = self.post_layernorm(pooled_output)

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        # always return dict for intermediate stage
        return {"hidden_states": hidden_states}

    @staticmethod
    def clip_vision_model_forward(
        self: CLIPVisionModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[list[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return CLIPVisionPipelineForwards.clip_vision_transformer_forward(
            self.vision_model,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )


def get_clip_naive_attention_forward():
    from transformers.models.clip.modeling_clip import CLIPAttention

    def forward(
        self: CLIPAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped

    return forward


def get_clip_flash_attention_forward():
    from colossalai.nn.layer.colo_attention import ColoAttention
    from transformers.models.clip.modeling_clip import CLIPAttention

    def forward(
        self: CLIPAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # [batch_size, tgt_len, embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [batch_size, tgt_len, embed_dim] -> [batch_size, tgt_len, num_heads, head_dim]
        query_states = query_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).contiguous()
        key_states = key_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).contiguous()
        value_states = value_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).contiguous()

        attention = ColoAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        attn_output = attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=attention_mask,
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, None

    return forward
