import functools
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalAudioAttention,
    Phi4MultimodalAudioModel,
    logger,
    simple_eager_attention_forward,
    unfold_tensor,
)

from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_flash_attention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class Phi4MultimodalForwards:
    @staticmethod
    def phi4_multimodal_audio_model_forward(
        self: Phi4MultimodalAudioModel,
        input_features: torch.FloatTensor,
        mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        shard_config: ShardConfig = None,
        **kwargs,
    ) -> torch.Tensor:
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

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        sp_rank = dist.get_rank(sp_group)

        if stage_manager is None or stage_manager.is_first_stage():
            hidden_states = self.encoder_embedding(input_features)

            if sp_mode == "ring_attn":
                # FIXME: Qwen2Vision does not have a batched input,
                # but they are concatenated. Current implementation
                # does not work even though somehow it passes the tests.
                # Need to support varlen input.
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

                # Recompute attention mask
                mask = torch.chunk(mask, sp_size, dim=1)[sp_rank]

            hidden_states, hs_mask, mask = self.forward_embeddings(hidden_states, mask)

        unfolded = False
        bs, seq_len, _ = hidden_states.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0

        if stage_manager is None or stage_manager.is_first_stage():
            if seq_len > max_seq_len:
                if chunk_pad_size > 0:
                    hidden_states_pad = F.pad(
                        hidden_states, (0, 0, 0, chunk_pad_size), "constant", 0
                    )
                    hidden_states = hidden_states_pad.to(hidden_states.device)

                hidden_states = unfold_tensor(hidden_states, max_seq_len)
                masks_unfold = None
                if mask is not None:
                    # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                    subsampled_pad_mask = mask.squeeze(
                        1
                    )  # [bz, subsampled_unmask_seq_len]
                    extra_padded_subsamlped_pad_mask = F.pad(
                        subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                    )  # extra padding to the pad mask
                    extra_padded_subsamlped_pad_mask = (
                        extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                    )
                    masks_unfold = unfold_tensor(
                        extra_padded_subsamlped_pad_mask, max_seq_len
                    )  # unfold the pad mask like we did to the input tensor
                    masks_unfold = masks_unfold.squeeze(
                        -1
                    ).bool()  # unfold op does not support bool tensor
                hs_mask = self.calculate_hs_mask(
                    hidden_states, hidden_states.device, masks_unfold
                )  # calculate hs_mask based on the unfolded pad mask

            relative_attention_bias = self.relative_attention_bias_layer(hidden_states)
            attention_mask = hs_mask.unsqueeze(1) + relative_attention_bias

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if (
            split_input
            and shard_config.enable_tensor_parallelism
            and attention_mask is not None
        ):
            # Recompute attention mask
            assert attention_mask.ndim == 4
            tp_size = shard_config.tensor_parallel_size
            tp_rank = dist.get_rank(shard_config.tensor_parallel_process_group)
            attention_mask = torch.chunk(attention_mask, tp_size, dim=1)[tp_rank]

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.encoders))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.encoders))

        for layer in self.encoders[start_idx:end_idx]:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)

        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
            }

        if unfolded is True:
            embed_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(bs, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                hidden_states = hidden_states[:, :-chunk_pad_size, :]

        return hidden_states


class Phi4MultimodalAudioAttentionForwards:
    @staticmethod
    def forward(
        self: Phi4MultimodalAudioAttention,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        shard_config: Optional[ShardConfig] = None,
        **kwargs,
    ) -> torch.Tensor:
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

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if sp_mode == "ring_attn":
            attention_interface: Callable = functools.partial(
                context_parallel_flash_attention, sp_group=sp_group
            )
        else:
            attention_interface: Callable = simple_eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
