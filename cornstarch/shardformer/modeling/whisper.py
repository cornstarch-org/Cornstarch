import functools
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.shardformer.shard import ShardConfig
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder,
    WhisperAttention,
)

from transformers.cache_utils import EncoderDecoderCache
from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_flash_attention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["ring_attn"]


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

        ContextParallelBatchSplitUtils.clear_cache()

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


class WhisperAttentionForwards:
    @staticmethod
    def forward(
        self: WhisperAttention,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        assert not is_cross_attention, "Cross attention is not supported yet"

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
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape)

        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, : key_states.shape[-2]]

        if sp_mode == "ring_attn":
            attention_interface: Callable = functools.partial(
                context_parallel_flash_attention, sp_group=sp_group
            )
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=causal_mask,
            dropout_p=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, past_key_value
