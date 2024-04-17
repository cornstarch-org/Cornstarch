from typing import Optional, Tuple, Union

import torch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers.utils import logging


class Dinov2PipelineForwards:
    """
    This class servers as a micro library for forward function substitution of Dinov2 models
    under pipeline setting.
    """

    @staticmethod
    def dinov2_model_forward(
        self: Dinov2Model,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[list[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        logger = logging.get_logger(__name__)

        # Preprocess passed in arguments
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

        if stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            embedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos
            )
            hidden_states = embedding_output
        else:
            assert (
                hidden_states is not None
            ), f"Current stage is {stage_manager.stage}, hidden_states should not be None"

        start_idx, end_idx = stage_index[0], stage_index[1]
        for idx, layer_module in enumerate(
            self.encoder.layer[start_idx:end_idx], start=start_idx
        ):
            layer_head_mask = head_mask[idx] if head_mask is not None else None

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

        if not stage_manager.is_last_stage():
            return {"hidden_states": hidden_states}

        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=None,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
