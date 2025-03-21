from typing import Optional, Tuple, Union

import torch
from colossalai.shardformer.shard import ShardConfig
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder


class WhisperModelForwards:
    @staticmethod
    def whisper_encoder_forward(
        self: WhisperEncoder,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.FloatTensor]] = (),
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

        stage_manager = shard_config.pipeline_stage_manager

        # Process inputs if at the first stage of encoder.
        if stage_manager is None or stage_manager.is_first_stage():
            inputs_embeds = nn.functional.gelu(self.conv1(input_features))
            inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

            inputs_embeds = inputs_embeds.permute(0, 2, 1)
            embed_pos = self.embed_positions.weight

            hidden_states = inputs_embeds + embed_pos
            hidden_states = nn.functional.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

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

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {"hidden_states": hidden_states}
            if output_hidden_states:
                outputs["encoder_states"] = encoder_states
            if output_attentions:
                outputs["all_attentions"] = all_attentions
            return outputs

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
