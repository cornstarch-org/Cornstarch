import functools
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from cornstarch.models.evaclip.modeling_evaclip import (
    EvaCLIPAttention,
    EvaCLIPVisionModel,
    EvaCLIPVisionTransformer,
    logger,
)
from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_flash_attention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["ring_attn"]


class EvaCLIPModelForwards:
    @staticmethod
    def eva_clip_vision_transformer_forward(
        self: EvaCLIPVisionTransformer,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            hidden_states = self.embeddings(pixel_values)

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size

        # Support SP + PP. Later stages have already received the split input.
        if sp_mode == "ring_attn":
            split_input = stage_manager is None or stage_manager.is_first_stage()
            if split_input:
                ContextParallelBatchSplitUtils.create_context_parallel_split(
                    # fake attention mask
                    torch.empty(hidden_states.shape[:2], device="meta"),
                    sp_group,
                    dist_mode=ContextParallelDistributionMode.UNIFORM,
                )

                hidden_states = ContextParallelBatchSplitUtils.split_batch(
                    hidden_states,
                    sp_group,
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
                    None,
                    None,
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

        ContextParallelBatchSplitUtils.clear_cache()

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

    @staticmethod
    def evaclip_vision_model_forward(
        self: EvaCLIPVisionModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        encoder_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_attentions: Optional[Tuple[torch.FloatTensor]] = (),
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return EvaCLIPModelForwards.eva_clip_vision_transformer_forward(
            self.vision_model,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hidden_states=hidden_states,
            encoder_states=encoder_states,
            all_attentions=all_attentions,
            shard_config=shard_config,
        )


class EvaCLIPAttentionForwards:
    @staticmethod
    def forward(
        self: EvaCLIPAttention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        shard_config: Optional[ShardConfig] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if shard_config is not None and shard_config.enable_sequence_parallelism:
            sp_mode: str = shard_config.sequence_parallelism_mode
            sp_size: int = shard_config.sequence_parallel_size
            sp_group: dist.ProcessGroup = shard_config.sequence_parallel_process_group

            assert (
                sp_mode in _SUPPORTED_CP_MODE
            ), f"SP mode {sp_mode} is not supported by {type(self)} yet"
            assert (
                sp_size > 1 and sp_group is not None
            ), "Must specify sp_size and sp_group for sequence parallel"
        else:
            sp_mode = None
            sp_size = None
            sp_group = None

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if sp_mode == "ring_attn":
            attention_interface: Callable = functools.partial(
                context_parallel_flash_attention, sp_group=sp_group
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            dropout_p=0.0 if not self.training else self.dropout,
            scaling=self.scale,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights
