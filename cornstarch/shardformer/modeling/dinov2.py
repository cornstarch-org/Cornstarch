import functools
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Encoder,
    Dinov2Model,
    Dinov2SelfAttention,
    eager_attention_forward,
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


class Dinov2ModelForwards:
    @staticmethod
    def dinov2_encoder_forward(
        self: Dinov2Encoder,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
    ) -> Union[tuple, BaseModelOutput]:
        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layer))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layer))

        for i, layer_module in enumerate(
            self.layer[start_idx:end_idx], start=start_idx
        ):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
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
                outputs["all_hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["all_self_attentions"] = all_self_attentions
            return outputs

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def dinov2_model_forward(
        self: Dinov2Model,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
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

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            embedding_output = self.embeddings(
                pixel_values, bool_masked_pos=bool_masked_pos
            )
            hidden_states = embedding_output

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input and sp_mode == "ring_attn":
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

        encoder_outputs = Dinov2ModelForwards.dinov2_encoder_forward(
            self.encoder,
            hidden_states=hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
        )

        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return encoder_outputs

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

    @staticmethod
    def dinov2_backbone_forward(
        self: Dinov2Backbone,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
    ) -> BackboneOutput:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            embedding_output = self.embeddings(pixel_values)
            hidden_states = embedding_output

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input and sp_mode == "ring_attn":
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

        outputs = Dinov2ModelForwards.dinov2_encoder_forward(
            self.encoder,
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
        )

        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return outputs

        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(
                        batch_size, height // patch_size, width // patch_size, -1
                    )
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


class Dinov2SelfAttentionForwards:
    @staticmethod
    def forward(
        self: Dinov2SelfAttention,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        shard_config: Optional[ShardConfig] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

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

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if sp_mode == "ring_attn":
            assert (
                head_mask is None
            ), "head_mask is not supported in context parallelism"
            attention_interface: Callable = functools.partial(
                context_parallel_flash_attention, sp_group=sp_group
            )
        else:
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and output_attentions:
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation
                    ]

        context_layer, _ = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask=head_mask,
            scaling=self.scaling,
            dropout_p=0.0 if not self.training else self.dropout_prob,
            **kwargs,
        )

        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None
