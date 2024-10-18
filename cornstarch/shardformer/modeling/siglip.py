from typing import Optional, Tuple, Union

import torch
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionModel,
    SiglipVisionTransformer,
)


class SiglipVisionModelForwards:
    @staticmethod
    def siglip_vision_transformer_forward(
        self: SiglipVisionTransformer,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.Tensor]] = (),
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

        if stage_manager is None or stage_manager.is_first_stage():
            hidden_states = self.embeddings(
                pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
            )

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.encoder.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.encoder.layers))

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

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {"hidden_states": hidden_states}
            if output_hidden_states:
                outputs["encoder_states"] = encoder_states
            if output_attentions:
                outputs["all_attentions"] = all_attentions
            return outputs

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

    @staticmethod
    def siglip_vision_model_forward(
        self: SiglipVisionModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        return SiglipVisionModelForwards.siglip_vision_transformer_forward(
            self.vision_model,
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict,
            interpolate_pos_encoding,
            hidden_states,
            encoder_states,
            all_attentions,
            shard_config,
        )
