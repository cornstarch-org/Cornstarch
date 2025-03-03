import functools
from typing import Callable, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.shardformer.layer import dist_cross_entropy
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    split_forward_gather_backward,
)
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.cache_utils import Cache, HybridCache
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2ForCausalLM,
    Gemma2Model,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)
from transformers.processing_utils import Unpack

from cornstarch.kernel.interface import BitfieldUtils
from cornstarch.shardformer.layers.context_parallel_bitfield_attention import (
    context_parallel_bitfield_attention_forward,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class Gemma2ModelForwards:
    @staticmethod
    def gemma2_model_forward(
        self: Gemma2Model,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.FloatTensor]] = (),
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
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
        if self.gradient_checkpointing and self.training and use_cache:
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
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds

            # normalized
            # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
            # See https://github.com/huggingface/transformers/pull/29402
            normalizer = torch.tensor(
                self.config.hidden_size**0.5, dtype=hidden_states.dtype
            )
            hidden_states = hidden_states * normalizer

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = hidden_states.shape
            past_key_values = HybridCache(
                self.config,
                max_batch_size=batch_size,
                max_cache_len=seq_len,
                dtype=inputs_embeds.dtype,
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

        if position_embeddings is None:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # This is needed to correctly slice the mask without data-dependent slicing later on if using dynamo tracing
        # (retrieving the same value from `cache_position` later on would crash dynamo)
        if last_cache_position is None:
            last_cache_position = 0
            if attention_mask is not None:
                # In case a 4d mask is passed directly without using `generate`, we have to rely on cache_position
                # It will break dynamo tracing but there are no way around it (and it should never happen in practice)
                last_cache_position = (
                    attention_mask.shape[-1]
                    if attention_mask.dim() == 2
                    else cache_position[-1].item()
                )

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        cp_dist_mode = getattr(
            shard_config,
            "context_parallel_distribution_mode",
            ContextParallelDistributionMode.UNIFORM,
        )

        if self.config._attn_implementation == "bitfield_attention":
            if BitfieldUtils.sequence_lengths_cache is None:
                sequence_lengths: torch.Tensor = torch.ne(attention_mask, 0).sum(dim=1)
                BitfieldUtils.set_sequence_lengths_cache(sequence_lengths.tolist())
            attn_mask = attention_mask
        else:
            # causal mask
            attn_mask = self._update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                output_attentions,
            )

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input:
            if sp_mode == "ring_attn":
                assert self.config._attn_implementation == "bitfield_attention", (
                    "Cornstarch context parallelism is only supported with bitfield_attention. "
                    f"Got {self.config._attn_implementation}"
                )
                sequence_lengths = BitfieldUtils.sequence_lengths_cache[0]
                hidden_states, _, offsets = ContextParallelBatchSplitUtils.split_batch(
                    hidden_states,
                    sequence_lengths,
                    attn_mask,
                    sp_group,
                    dist_mode=cp_dist_mode,
                )  # shape: [B, L // sp_size, ...]
                position_ids = offsets
            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size
                )

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

        kwargs = {}
        if self.config._attn_implementation == "bitfield_attention":
            seqlen_ks, offsets = BitfieldUtils.get_sequence_lengths_cache()
            seqlen_qs = seqlen_ks
            if sp_mode == "ring_attn":
                seqlen_qs, offsets = (
                    ContextParallelBatchSplitUtils.get_context_parallel_sequence_lengths_cache()
                )
                indices_perm, indices_inverse_perm = (
                    ContextParallelBatchSplitUtils.get_permutate_cache()
                )
                kwargs["indices_perm"] = indices_perm
                kwargs["indices_inverse_perm"] = indices_inverse_perm

            kwargs["seqlen_qs"] = seqlen_qs
            kwargs["seqlen_ks"] = seqlen_ks
            kwargs["offsets"] = offsets

        # decoder layers
        for decoder_layer in self.layers[start_idx:end_idx]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attn_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    last_cache_position,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    last_cache_position=last_cache_position,
                    **flash_attn_kwargs,
                    **kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)

        BitfieldUtils.clear_cache()
        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {
                "hidden_states": hidden_states,
                "cache_position": cache_position,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        return output if return_dict else output.to_tuple()

    def gemma2_for_causal_lm_forward(
        self: Gemma2ForCausalLM,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.FloatTensor]] = (),
        shard_config: ShardConfig = None,
        **loss_kwargs,
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

        if (
            shard_config.sequence_parallelism_mode == "ring_attn"
            and shard_config.parallel_output
        ):
            # Split labels too
            sp_group = shard_config.sequence_parallel_process_group
            cp_dist_mode = getattr(
                shard_config,
                "context_parallel_distribution_mode",
                ContextParallelDistributionMode.UNIFORM,
            )

            assert self.config._attn_implementation == "bitfield_attention", (
                "Cornstarch context parallelism is only supported with bitfield_attention. "
                f"Got {self.config._attn_implementation}"
            )

            sequence_lengths: torch.Tensor = torch.ne(attention_mask, 0).sum(dim=1)
            labels, _, _ = ContextParallelBatchSplitUtils.split_batch(
                labels,
                sequence_lengths,
                attention_mask,
                sp_group,
                dist_mode=cp_dist_mode,
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = Gemma2ModelForwards.gemma2_model_forward(
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
            position_embeddings=position_embeddings,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
            force_sp_gather=False,
            **loss_kwargs,
        )

        BitfieldUtils.clear_cache()
        ContextParallelBatchSplitUtils.clear_cache()

        stage_manager = shard_config.pipeline_stage_manager
        if not (stage_manager is None or stage_manager.is_last_stage()):
            return outputs

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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


class Gemma2AttentionForwards:
    @staticmethod
    def forward(
        self: Gemma2Attention,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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

        # sp: all-to-all communication when introducing ulysses context parallelism
        if sp_mode == "all_to_all":
            query_states = all_to_all_comm(query_states, sp_group)
            key_states = all_to_all_comm(key_states, sp_group)
            value_states = all_to_all_comm(value_states, sp_group)
            input_shape[1] = hidden_shape[1] = query_states.shape[1]

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "sliding_window": self.sliding_window,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

            # Here we need to slice as we use a static cache by default, but FA2 does not support it
            if (
                attention_mask is not None
                and self.config._attn_implementation == "flash_attention_2"
            ):
                seq_len = attention_mask.shape[-1]
                key_states, value_states = (
                    key_states[:, :, :seq_len, :],
                    value_states[:, :, :seq_len, :],
                )

        if sp_mode == "ring_attn":
            assert self.config._attn_implementation == "bitfield_attention", (
                "Cornstarch context parallelism is only supported with bitfield_attention. "
                f"Got {self.config._attn_implementation}"
            )

            attention_interface: Callable = functools.partial(
                context_parallel_bitfield_attention_forward, sp_group=sp_group
            )
        else:
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get(
                    "output_attentions", False
                ):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[
                        self.config._attn_implementation
                    ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # sp: all-to-all communication when introducing ulysses context parallelism
        if sp_mode == "all_to_all":
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2
            )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
