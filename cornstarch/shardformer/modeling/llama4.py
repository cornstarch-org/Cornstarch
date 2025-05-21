import functools
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    split_forward_gather_backward,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextModel,
    Llama4TextMoe,
    Llama4ForCausalLM,
    FlashAttentionKwargs,
    Llama4TextAttention,
    apply_rotary_emb,
    eager_attention_forward,
    logger,
)
from colossalai.shardformer.layer import dist_cross_entropy
from transformers.cache_utils import Cache, HybridChunkedCache
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from cornstarch.kernel.bitfield_attention import BitfieldUtils
from cornstarch.shardformer.layers.context_parallel_bitfield_attention import (
    context_parallel_bitfield_attention_forward,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class Llama4ModelForwards:
    @staticmethod
    def llama4_text_model_forward(
        self: Llama4TextModel,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        freq_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross entropy
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
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
                    "You must specify exactly one of input_ids or inputs_embeds"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(
                    input_ids.to(self.embed_tokens.weight.device)
                )

            hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = HybridChunkedCache(
                self.config, inputs_embeds.shape[0], inputs_embeds.shape[1]
            )

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if stage_manager is None or stage_manager.is_first_stage():
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # create position embeddings to be shared across the decoder layers
            freq_cis = self.rotary_emb(hidden_states, position_ids)

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        cp_dist_mode = getattr(
            shard_config,
            "context_parallel_distribution_mode",
            ContextParallelDistributionMode.UNIFORM,
        )

        if self.config._attn_implementation == "bitfield_attention":
            causal_mask = attention_mask
            chunk_causal_mask = None
        else:
            causal_mask, chunk_causal_mask = self._update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                output_attentions,
                use_cache=use_cache,
            )

        # Support SP + PP. Later stages have already received the split input.
        split_input = stage_manager is None or stage_manager.is_first_stage()
        if split_input:
            if sp_mode == "ring_attn":
                assert self.config._attn_implementation == "bitfield_attention", (
                    "Cornstarch context parallelism is only supported with cornstarch attention. "
                    f"Got {self.config._attn_implementation}"
                )

                ContextParallelBatchSplitUtils.create_context_parallel_split(
                    attention_mask, sp_group, dist_mode=cp_dist_mode
                )
                offsets_per_rank = (
                    ContextParallelBatchSplitUtils.get_context_parallel_offsets_cache()
                )

                hidden_states = ContextParallelBatchSplitUtils.split_batch(
                    hidden_states,
                    sp_group,
                )
                position_ids = (
                    ContextParallelBatchSplitUtils.get_context_parallel_offsets_cache(
                        dist.get_rank(sp_group)
                    ).unsqueeze(0)
                )
            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size
                )

            # Recompute freq_cis
            freq_cis = self.rotary_emb(hidden_states, position_ids)

        ContextParallelBatchSplitUtils.set_context_parallel_offsets_cache(
            offsets_per_rank
        )

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

        kwargs = {}
        if sp_mode == "ring_attn":
            kwargs.update(
                {
                    "compressed_mask": ContextParallelBatchSplitUtils.get_local_compressed_mask(
                        causal_mask, sp_group
                    ),
                    "offsets_per_rank": offsets_per_rank,
                }
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for decoder_layer in self.layers[start_idx:end_idx]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    chunk_causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    False,  # output_router_logits is False
                    use_cache,
                    cache_position,
                    freq_cis,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    chunk_causal_mask=chunk_causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=freq_cis,
                    **flash_attn_kwargs,
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
                "freq_cis": freq_cis,
            }
            if output_hidden_states:
                outputs["all_hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["all_self_attentions"] = all_self_attentions
            outputs.update(kwargs)
            return outputs

        hidden_states = self.norm(hidden_states)
        if shard_config.enable_sequence_parallelism and (
            (not shard_config.parallel_output) or force_sp_gather
        ):
            hidden_states = gather_sp_output(hidden_states, shard_config)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def llama4_for_causal_lm_forward(
        self: Llama4ForCausalLM,
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
        logits_to_keep: Union[int, torch.Tensor] = 0,
        freq_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
        **kwargs,
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

            if offsets_per_rank is None:
                # This is the first stage. Create offsets
                assert stage_manager is None or stage_manager.is_first_stage()
                ContextParallelBatchSplitUtils.create_context_parallel_split(
                    attention_mask, sp_group, dist_mode=cp_dist_mode
                )
            else:
                # Set given offsets cache to batch split utils
                ContextParallelBatchSplitUtils.set_context_parallel_offsets_cache(
                    offsets_per_rank
                )

            labels = ContextParallelBatchSplitUtils.split_batch(
                labels, sp_group, is_label=True
            )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = Llama4ModelForwards.llama4_text_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            freq_cis=freq_cis,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
            force_sp_gather=False,
            offsets_per_rank=offsets_per_rank,
            **kwargs,
        )
        past_key_values = None

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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Llama4AttentionForwards:
    @staticmethod
    def forward(
        self: Llama4TextAttention,
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

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        if self.use_rope:  # the 16E model skips rope for long context on certain layers
            query_states, key_states = apply_rotary_emb(
                query_states, key_states, position_embeddings.to(query_states.device)
            )

        if hasattr(self, "qk_norm"):  # the 128E model does not use qk_norm
            query_states = self.qk_norm(query_states)
            key_states = self.qk_norm(key_states)

        # Use temperature tuning from https://arxiv.org/abs/2501.19399) to NoROPE layers
        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = (
                torch.log(
                    torch.floor((cache_position.float() + 1.0) / self.floor_scale) + 1.0
                )
                * self.attn_scale
                + 1.0
            )
            attn_scales = attn_scales.view((1, input_shape[-1], 1, 1)).expand(
                (*input_shape, 1, 1)
            )  # batch size > 1
            query_states = (query_states * attn_scales).to(query_states.dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
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

        # sp: all-to-all communication when introducing ulysses context parallelism
        if sp_mode == "all_to_all":
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Llama4TextMoeForwards:
    @staticmethod
    def forward(
        self: Llama4TextMoe, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = self.router(hidden_states).transpose(0, 1)
        tokens_per_expert = batch * seq_len

        router_top_value, router_indices = torch.topk(
            router_logits.transpose(0, 1), self.top_k, dim=1
        )
        router_scores = (
            torch.full_like(router_logits.transpose(0, 1), float("-inf"))
            .scatter_(1, router_indices, router_top_value)
            .transpose(0, 1)
        )
        # We do this to make sure we have -inf for non topK tokens before going through the !
        # Here we are just creating a tensor to index each and every single one of the hidden states. Let s maybe register a buffer for this!
        router_indices = (
            torch.arange(tokens_per_expert, device=hidden_states.device)
            .view(1, -1)
            .expand(router_scores.size(0), -1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
        routed_in = torch.gather(
            input=hidden_states,
            dim=0,
            index=router_indices,
        ).to(hidden_states.device)
        # we gather inputs corresponding to each expert based on the router indices
        routed_in = routed_in * router_scores.reshape(-1, 1)
        routed_out = self.experts(routed_in)

        # If tensor parallelism is enabled, `out` is the result of _reduceForward.apply(),
        # which is a view of a tensor.
        # Modifying `out` in-place occurs an error, thus clone this tensor.
        out = self.shared_expert(hidden_states).clone()

        out.scatter_add_(
            dim=0, index=router_indices, src=routed_out.view(-1, hidden_dim)
        )
        return out, router_scores
