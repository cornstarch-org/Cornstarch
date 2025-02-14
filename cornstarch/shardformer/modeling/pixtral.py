from typing import Optional, Tuple, Union

import torch
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionModel,
    generate_block_attention_mask,
    logger,
    position_ids_in_meshgrid,
)


class PixtralVisionModelForwards:
    @staticmethod
    def pixtral_vision_model_forward(
        self: PixtralVisionModel,
        pixel_values: Optional[list[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.FloatTensor]] = (),
        shard_config: ShardConfig = None,
        *args,
        **kwargs,
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

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # pass images through initial convolution independently
            patch_embeds_list = [
                self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in pixel_values
            ]

            # flatten to a single sequence
            patch_embeds = torch.cat(
                [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1
            )
            patch_embeds = self.ln_pre(patch_embeds)

            # positional embeddings
            position_ids = position_ids_in_meshgrid(
                patch_embeds_list,
                max_width=self.config.image_size // self.config.patch_size,
            ).to(self.device)

            position_embeddings = self.patch_positional_embedding(
                patch_embeds, position_ids
            )

            attention_mask = generate_block_attention_mask(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
            )

            hidden_states = patch_embeds

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(
                len(self.transformer.layers)
            )
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.transformer.layers))

        for encoder_layer in self.transformer.layers[start_idx:end_idx]:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.transformer.gradient_checkpointing and self.training:
                layer_outputs = self.transformer._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {
                "hidden_states": hidden_states,
                "position_embeddings": position_embeddings,
                "attention_mask": attention_mask,
            }
            if output_hidden_states:
                outputs["encoder_states"] = encoder_states
            if output_attentions:
                outputs["all_attentions"] = all_attentions
            return outputs

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
