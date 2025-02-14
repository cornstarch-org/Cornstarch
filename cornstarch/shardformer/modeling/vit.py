from typing import Optional, Tuple, Union

import torch
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.models.vit.modeling_vit import (
    ViTModel,
    logger,
)


class ViTModelForwards:
    @staticmethod
    def vit_model_forward(
        self: ViTModel,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.FloatTensor]] = (),
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

        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is not None:
            if output_attentions:
                logger.warning_once(
                    "output_attentions=True is not supported for pipeline models at the moment."
                )
                output_attentions = False
            if output_hidden_states:
                logger.warning_once(
                    "output_hidden_states=True is not supported for pipeline models at the moment."
                )
                output_hidden_states = False

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
            expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
            if pixel_values.dtype != expected_dtype:
                pixel_values = pixel_values.to(expected_dtype)

            embedding_output = self.embeddings(
                pixel_values,
                bool_masked_pos=bool_masked_pos,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            hidden_states = embedding_output

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.encoder.layer))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.encoder.layer))

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.encoder.layer[start_idx:end_idx]):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.encoder.gradient_checkpointing and self.training:
                layer_outputs = self.encoder._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask, output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {"hidden_states": hidden_states}
            if output_hidden_states:
                outputs["encoder_states"] = encoder_states
            if output_attentions:
                outputs["attentions"] = all_attentions
            return outputs

        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# class ViTAttentionForwards:
#     @staticmethod
#     def sdpa_forward(
#         self: ViTSdpaSelfAttention,
#         hidden_states: torch.FloatTensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         if output_attentions or head_mask is not None:
#             logger.warning_once(
#                 "`ViTSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
#                 "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
#                 "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
#                 'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
#             )
#             return ViTAttentionForwards.forward(
#                 self, hidden_states, head_mask, output_attentions
#             )


#     @staticmethod
#     def forward(
#         self: ViTSelfAttention,
#         hidden_states: torch.Tensor,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: bool = False,
#     ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
#         pass
