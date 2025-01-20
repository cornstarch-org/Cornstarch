import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.shardformer.layer import dist_cross_entropy
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    split_forward_gather_backward,
)
from colossalai.shardformer.shard.shard_config import ShardConfig
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from cornstarch.kernel.interface import convert_bit_attention_mask_to_block_mask
from cornstarch.models.internlm2.modeling_internlm2 import (
    InternLM2Attention,
    InternLM2ForCausalLM,
    InternLM2Model,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)
from cornstarch.shardformer.layers.utils import (
    repeat_attention_mask_heads,
    ContextParallelBatchUtils,
)

flex_attention = torch.compile(flex_attention, fullgraph=True)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class InternLM2ModelForwards:
    @staticmethod
    def internlm2_model_forward(
        self: InternLM2Model,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross entropy
    ) -> Union[Tuple, BaseModelOutputWithPast]:
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and self.gradient_checkpointing and self.training:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is not None:
            if use_cache:
                logger.warning_once(
                    "use_cache=True is not supported for pipeline models at the moment."
                )
                use_cache = False

        if stage_manager is None or stage_manager.is_first_stage():
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.tok_embeddings(input_ids)

            hidden_states = inputs_embeds

        return_legacy_cache = False
        if use_cache and not isinstance(
            past_key_values, Cache
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        ring_attn_mode = getattr(
            shard_config, "ring_attention_distribution_mode", "uniform"
        )

        if sp_mode == "ring_attn":
            # Do nothing. Our RingAttention implementation converts attention mask automatically.
            attn_mask = attention_mask
        else:
            # non ring attention (either all-to-all or non sequence parallel)
            if attention_mask.ndim == 2:
                if getattr(attention_mask, "cornstarch_is_bitattention", False):
                    # BitAttentionMask for anymask
                    num_heads = (
                        self.config.num_attention_heads
                        // shard_config.tensor_parallel_size
                    )
                    attn_mask = convert_bit_attention_mask_to_block_mask(
                        attention_mask, num_heads
                    )
                else:
                    # causal mask
                    attn_mask = self._update_causal_mask(
                        attention_mask,
                        hidden_states,
                        cache_position,
                        past_key_values,
                        output_attentions,
                    )
            elif attention_mask.ndim == 3:  # anymask
                assert (
                    self.config._attn_implementation != "flash_attention_2"
                ), "Flash Attention 2 does not support AnyMask. Use either `sdpa` or `eager`."
                num_heads = (
                    self.config.num_attention_heads // shard_config.tensor_parallel_size
                )
                attn_mask = repeat_attention_mask_heads(attention_mask, num_heads)
            else:
                raise ValueError(
                    f"Unsupported attention mask dimension: {attention_mask.ndim}"
                )

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input:
            # Ring Attention batch processing
            if sp_mode == "ring_attn":
                hidden_states = ContextParallelBatchUtils.split_batch(
                    hidden_states, sp_group, seq_dim=1, ring_attn_mode=ring_attn_mode
                )  # shape: [B, L // sp_size, ...]
                position_ids = ContextParallelBatchUtils.split_batch(
                    position_ids, sp_group, seq_dim=1, ring_attn_mode=ring_attn_mode
                )  # shape: [B, L // sp_size]

            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size
                )

        # decoder layers
        next_decoder_cache = None

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

        for decoder_layer in self.layers[start_idx:end_idx]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attn_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {
                "hidden_states": hidden_states,
                "cache_position": cache_position,
                "position_ids": position_ids,
            }
            if output_hidden_states:
                outputs["all_hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["all_self_attentions"] = all_self_attentions
            return outputs

        hidden_states = self.norm(hidden_states)
        if shard_config.enable_sequence_parallelism and (
            (not shard_config.parallel_output) or force_sp_gather
        ):
            hidden_states = gather_sp_output(hidden_states, shard_config)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        # Clear cache so that it is not used in the next forward pass
        ContextParallelBatchUtils.clear_split_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    @staticmethod
    def internlm2_for_causal_lm_forward(
        self: InternLM2ForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        if (
            shard_config.sequence_parallelism_mode == "ring_attn"
            and shard_config.parallel_output
        ):
            # Split labels too
            sp_group = shard_config.sequence_parallel_process_group
            ring_attn_mode = getattr(
                shard_config, "ring_attention_distribution_mode", "uniform"
            )

            labels = ContextParallelBatchUtils.split_batch(
                labels,
                sp_group,
                seq_dim=1,
                ring_attn_mode=ring_attn_mode,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = InternLM2ModelForwards.internlm2_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
            force_sp_gather=False,
        )

        stage_manager = shard_config.pipeline_stage_manager
        if not (stage_manager is None or stage_manager.is_last_stage()):
            return outputs

        hidden_states = outputs[0]
        logits = self.output(hidden_states).float()

        loss = None
        if labels is not None:
            loss = dist_cross_entropy(
                labels,
                logits,
                shard_config,
                self.output.out_features,
                self.model.dtype,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class InternLM2AttentionForwards:
    @staticmethod
    def forward(
        self: InternLM2Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,  # pylint: disable=unused-argument
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at "
                "https://github.com/huggingface/transformers"
            )

        assert (
            self.config.pretraining_tp == 1
        ), "Support for model pretrained with tp is removed."
        output_attentions = False

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

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.wqkv(hidden_states)
        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=2 + self.num_key_value_groups,
            d=self.head_dim,
        )

        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]

        if sp_mode == "all_to_all":
            # This is quite inefficient rearrange, but it is needed to make the all_to_all_comm work
            query_states = rearrange(query_states, "b q h d -> b q (h d)")
            key_states = rearrange(key_states, "b q h d -> b q (h d)")
            value_states = rearrange(value_states, "b q h d -> b q (h d)")

            query_states = all_to_all_comm(query_states, sp_group)
            key_states = all_to_all_comm(key_states, sp_group)
            value_states = all_to_all_comm(value_states, sp_group)
            bsz, q_len, _ = query_states.size()

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
            key_states = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
            value_states = value_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # ==========
        # Common qkv computation part is done, now we can call the specific attention function
        # ==========

        attn_weights = None
        if sp_mode == "ring_attn":
            raise NotImplementedError("Ring Attention is not supported yet.")
        elif isinstance(attention_mask, BlockMask):
            attn_output, attn_weights = flex_attention(
                query_states,
                key_states,
                value_states,
                block_mask=attention_mask,
                enable_gqa=True,
                return_lse=True,
                kernel_options={
                    "BLOCK_M": 64,
                    "BLOCK_N": 64,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 64,
                    "BLOCK_M2": 64,
                    "BLOCK_N2": 32,
                },
            )
        elif self.config.attn_implementation == "flash_attention_2":
            # TODO: These transpose are quite inefficient but Flash Attention requires the layout
            # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=0.0,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        elif self.config.attn_implementation == "sdpa":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with
            # custom attn_mask, Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of
            # an inline conditional assignment in SDPA to support both torch.compile's dynamic shapes and full graph
            # options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = bool(causal_mask is None and q_len > 1)

            attn_output = torch.nn.functional.scaled_dot_product_attention(  # pylint: disable=E1102
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        if sp_mode == "all_to_all":
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2
            )
        attn_output = self.wo(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
