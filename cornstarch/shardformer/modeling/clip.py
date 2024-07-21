from typing import Optional, Tuple, Union

import torch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention
from colossalai.shardformer.shard.shard_config import ShardConfig
from torch import nn
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPVisionModel,
    CLIPVisionTransformer,
    logger,
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
        for encoder_layer in self.encoder.layers[start_idx:end_idx]:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.encoder.gradient_checkpointing and self.training:
                layer_outputs = self.encoder._gradient_checkpointing_func(
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


class CLIPForwards:
    def clip_flash_attention_forward(
        self: CLIPAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "CLIPModel is using ClipSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            output_attentions = False

        bsz, tgt_len, _ = hidden_states.size()

        # Code borrowed from:
        # https://github.com/huggingface/transformers/pull/30390/files#diff-7f53db5caa73a4cbeb0dca3b396e3d52f30f025b8c48d4daf51eb7abb6e2b949

        # [batch_size, tgt_len, embed_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # [batch_size, tgt_len, embed_dim] -> [batch_size, tgt_len, num_heads, head_dim]
        query_states = query_states.view(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value_states = value_states.view(
            bsz, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_mask = (
            causal_attention_mask
            if causal_attention_mask is not None
            else attention_mask
        )
        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attn_mask + attention_mask

        # CLIP text model uses both  `causal_attention_mask` and `attention_mask` sequentially.
        attn_output = ColoAttention.attention(
            query_states,
            key_states,
            value_states,
            attention_mask=(
                attn_mask if causal_attention_mask is not None else attention_mask
            ),
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None

    def clip_eager_forward(
        self: CLIPAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Copied from https://github.com/huggingface/transformers/blob/v4.40.2/src/transformers/models/clip/modeling_clip.py,
        with the following changes:
        - Use `self.embed_dim` instead of `self.hidden_size` to get the size of the hidden dimension.
          This is required for tensor parallelism.
        """
        # Input shape: Batch x Time x Channel
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
