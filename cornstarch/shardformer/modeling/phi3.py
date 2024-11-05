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
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3ForCausalLM,
    Phi3Model,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)

from cornstarch.shardformer.layers.ring_attention_anymask import RingAttentionAnyMask
from cornstarch.shardformer.layers.utils import repeat_attention_mask_heads

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class Phi3ModelForwards:
    @staticmethod
    def phi3_model_forward(
        self: Phi3Model,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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

        if stage_manager is None or stage_manager.is_first_stage():
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

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
            # Cornstarch uniform, zigzag, and random ring attention
            if attention_mask.ndim == 2:  # causal mask
                # Manually create a 3D causal mask [B, L, L] for ring attention
                # as transformers attention mask utility
                # may return None to let attention code
                # to automatically generate a causal mask inside.

                # Assume attention mask is a simple all-1 matrix.
                assert attention_mask.bool().all()
                batch_size, seq_len, _ = hidden_states.shape
                attention_mask = torch.tril(
                    torch.ones(
                        batch_size,
                        seq_len,
                        seq_len,
                        device=hidden_states.device,
                        dtype=torch.bool,
                    )
                )

            assert (
                attention_mask.ndim == 3
            ), f"Unsupported attention mask dimension: {attention_mask.ndim}"

            num_heads = self.config.num_attention_heads
            # shape: [B, H, L, L]
            attn_mask = repeat_attention_mask_heads(attention_mask, num_heads)

            # shape: [B, H, L // sp_size, L]
            attn_mask = RingAttentionAnyMask.split_batch(
                attn_mask, sp_group, seq_dim=2, ring_attn_mode=ring_attn_mode
            )
        else:
            # non ring attention (either all-to-all or non sequence parallel)
            if attention_mask.ndim == 2:  # causal mask
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
                num_heads = self.config.num_attention_heads
                attn_mask = repeat_attention_mask_heads(attention_mask, num_heads)
            else:
                raise ValueError(
                    f"Unsupported attention mask dimension: {attention_mask.ndim}"
                )

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input:
            if sp_mode == "ring_attn":
                hidden_states = RingAttentionAnyMask.split_batch(
                    hidden_states, sp_group, seq_dim=1, ring_attn_mode=ring_attn_mode
                )  # shape: [B, L // sp_size, ...]
                position_ids = RingAttentionAnyMask.split_batch(
                    position_ids, sp_group, seq_dim=1, ring_attn_mode=ring_attn_mode
                )  # shape: [B, L // sp_size]

            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size
                )

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
                outputs["hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["attentions"] = all_self_attentions
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
        RingAttentionAnyMask.clear_split_random_cache()

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
    def phi3_for_causal_lm_forward(
        self: Phi3ForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
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

            labels = RingAttentionAnyMask.split_batch(
                labels,
                sp_group,
                seq_dim=1,
                ring_attn_mode=ring_attn_mode,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = Phi3ModelForwards.phi3_model_forward(
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
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()

            loss = dist_cross_entropy(
                labels,
                logits,
                shard_config,
                self.lm_head.out_features,
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


class Phi3AttentionForwards:
    @staticmethod
    def forward(
        self: Phi3Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                "You are not running the flash-attention implementation, expect numerical differences."
            )

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

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        num_key_value_heads = self.num_key_value_heads

        if sp_mode == "all_to_all":
            query_pos *= sp_size
            num_key_value_heads *= sp_size

        query_states = qkv[..., :query_pos]
        key_states = qkv[
            ..., query_pos : query_pos + num_key_value_heads * self.head_dim
        ]
        value_states = qkv[..., query_pos + num_key_value_heads * self.head_dim :]

        # sp: all-to-all communication when introducing ulysses context parallelism
        if sp_mode == "all_to_all":
            query_states = all_to_all_comm(query_states, sp_group)
            key_states = all_to_all_comm(key_states, sp_group)
            value_states = all_to_all_comm(value_states, sp_group)
            bsz, q_len, _ = query_states.size()

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = (
            max(kv_seq_len, position_ids[:, -1].max().item() + 1)
            if position_ids is not None
            else kv_seq_len
        )
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

        # ==========
        # Common qkv computation part is done, now we can call the specific attention function
        # ==========

        attn_weights = None
        if sp_mode == "ring_attn":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = RingAttentionAnyMask.attention(
                query_states,
                key_states,
                value_states,
                sp_group,
                attention_mask,
                return_softmax=False,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        elif self.config._attn_implementation == "flash_attention_2":
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_dropout = self.attention_dropout if self.training else 0.0

            # Reashape to the expected shape for Flash Attention
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=attn_dropout,
                sliding_window=getattr(self.config, "sliding_window", None),
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
                is_causal=self.is_causal,
            )
        elif self.config._attn_implementation == "sdpa":
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and attention_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights += causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(value_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )

            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

        # sp: all-to-all communication when introducing ulysses context parallelism
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if sp_mode == "all_to_all":
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2
            )

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
