import functools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.shardformer.layer import dist_cross_entropy
from colossalai.shardformer.layer.loss import cross_entropy_1d
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    split_forward_gather_backward,
)
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.cache_utils import Cache, DynamicCache
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, _get_unpad_data
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaAttention,
    LlamaForCausalLM,
    LlamaModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    logger,
)
from transformers.processing_utils import Unpack

from cornstarch.kernel.bitfield_attention import BitfieldUtils
from cornstarch.shardformer.layers.context_parallel_bitfield_attention import (
    context_parallel_bitfield_attention_forward,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


class LlamaModelForwards:
    @staticmethod
    def llama_model_forward(
        self: LlamaModel,
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross entropy
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
        packed_seq_indices: Optional[torch.Tensor] = None,
        packed_seq_cu_seqlens: Optional[torch.Tensor] = None,
        packed_seq_max_seqlen: Optional[int] = None,
        packed_seq_shape: Optional[Tuple[int, int]] = None,
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
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            )

        if stage_manager is None or stage_manager.is_first_stage():
            position_ids = cache_position.unsqueeze(0)

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        cp_dist_mode = getattr(
            shard_config,
            "context_parallel_distribution_mode",
            ContextParallelDistributionMode.UNIFORM,
        )

        if self.config._attn_implementation == "bitfield_attention":
            attn_mask = attention_mask
        elif packed_seq_indices is not None and not (
            stage_manager is None or stage_manager.is_first_stage()
        ):
            # Non-first PP stage with already-packed hidden_states: skip _update_causal_mask
            # because the tensor shape is (total_tokens, hidden_dim), not (batch, seq, hidden_dim).
            # The packed attention path sets attn_mask=None anyway.
            attn_mask = None
        else:
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

            # Recompute position embeddings
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        ContextParallelBatchSplitUtils.set_context_parallel_offsets_cache(
            offsets_per_rank
        )

        # Packed sequence path: pack hidden_states once before the layer loop so all N layers
        # (attention + MLP) and lm_head operate on valid tokens only.
        if (
            sp_mode is None
            and self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and not use_cache
            and (stage_manager is None or stage_manager.is_first_stage())
        ):
            batch_size, seq_len = hidden_states.shape[:2]
            hidden_dim = hidden_states.shape[-1]
            packed_seq_indices, packed_seq_cu_seqlens, packed_seq_max_seqlen = (
                _get_unpad_data(attention_mask)
            )
            packed_seq_shape = (batch_size, seq_len)
            hidden_states = index_first_axis(
                hidden_states.view(batch_size * seq_len, hidden_dim), packed_seq_indices
            )
            cos, sin = position_embeddings
            cos = cos.expand(batch_size, seq_len, -1).reshape(
                batch_size * seq_len, -1
            )[packed_seq_indices]
            sin = sin.expand(batch_size, seq_len, -1).reshape(
                batch_size * seq_len, -1
            )[packed_seq_indices]
            position_embeddings = (cos, sin)
            flash_attn_kwargs["cu_seq_lens_q"] = packed_seq_cu_seqlens
            flash_attn_kwargs["cu_seq_lens_k"] = packed_seq_cu_seqlens
            flash_attn_kwargs["max_length_q"] = packed_seq_max_seqlen
            flash_attn_kwargs["max_length_k"] = packed_seq_max_seqlen
            attn_mask = None
        elif packed_seq_indices is not None:
            # Non-first PP stage: hidden_states and position_embeddings are already packed;
            # inject the pre-computed cu_seqlens into flash_attn_kwargs.
            flash_attn_kwargs["cu_seq_lens_q"] = packed_seq_cu_seqlens
            flash_attn_kwargs["cu_seq_lens_k"] = packed_seq_cu_seqlens
            flash_attn_kwargs["max_length_q"] = packed_seq_max_seqlen
            flash_attn_kwargs["max_length_k"] = packed_seq_max_seqlen
            attn_mask = None

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
                        attn_mask, sp_group
                    ),
                    "offsets_per_rank": offsets_per_rank,
                }
            )
        # Pass flash_attn_kwargs to decoder layers so cu_seq_lens_q/k and max_length_q/k
        # (injected above for the packed path) reach LlamaAttentionForwards.forward.
        kwargs.update(flash_attn_kwargs)

        # Clear any stale compressed_mask cache left by a previous microbatch's
        # gradient-checkpoint recomputation.  Each new forward pass must build a
        # fresh cache; the end-of-forward clear_cache() call below does not run
        # during recomputation (only decoder_layer.__call__ is re-invoked), so
        # without this guard the wrong per-microbatch mask is returned and the
        # checkpoint shape-check fails on backward.
        BitfieldUtils.clear_cache()

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
                    position_embeddings,
                    **kwargs,
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
                    position_embeddings=position_embeddings,
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
                "position_embeddings": position_embeddings,
            }
            if output_hidden_states:
                outputs["all_hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["all_self_attentions"] = all_self_attentions
            if packed_seq_indices is not None:
                outputs["packed_seq_indices"] = packed_seq_indices
                outputs["packed_seq_cu_seqlens"] = packed_seq_cu_seqlens
                outputs["packed_seq_max_seqlen"] = packed_seq_max_seqlen
                outputs["packed_seq_shape"] = packed_seq_shape
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

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        return output if return_dict else output.to_tuple()

    @staticmethod
    def llama_for_causal_lm_forward(
        self: LlamaForCausalLM,
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
        packed_seq_indices: Optional[torch.Tensor] = None,
        packed_seq_cu_seqlens: Optional[torch.Tensor] = None,
        packed_seq_max_seqlen: Optional[int] = None,
        packed_seq_shape: Optional[Tuple[int, int]] = None,
        **kwargs: Unpack[KwargsForCausalLM],
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

        # Determine if packed sequence path will be used.
        # Resolve use_cache the same way llama_model_forward does so the condition matches.
        resolved_use_cache = use_cache if use_cache is not None else self.config.use_cache
        if resolved_use_cache and self.model.gradient_checkpointing and self.training:
            resolved_use_cache = False
        sp_mode = shard_config.sequence_parallelism_mode
        if (
            packed_seq_indices is None  # not already computed by a prior PP stage
            and sp_mode is None
            and self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and not resolved_use_cache
            and (stage_manager is None or stage_manager.is_first_stage())
        ):
            packed_seq_indices, packed_seq_cu_seqlens, packed_seq_max_seqlen = (
                _get_unpad_data(attention_mask)
            )
            packed_seq_shape = (attention_mask.shape[0], attention_mask.shape[1])

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
        outputs = LlamaModelForwards.llama_model_forward(
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
            offsets_per_rank=offsets_per_rank,
            packed_seq_indices=packed_seq_indices,
            packed_seq_cu_seqlens=packed_seq_cu_seqlens,
            packed_seq_max_seqlen=packed_seq_max_seqlen,
            packed_seq_shape=packed_seq_shape,
            **kwargs,
        )
        past_key_values = None

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return outputs

        # For PP last stage, packed_seq params arrive in the outputs dict.
        if isinstance(outputs, dict):
            packed_seq_indices = outputs.get("packed_seq_indices", packed_seq_indices)
            packed_seq_shape = outputs.get("packed_seq_shape", packed_seq_shape)

        hidden_states = outputs[0]

        loss = None
        if packed_seq_indices is not None:
            # Packed sequence path: hidden_states is (total_tokens, hidden_dim).
            # lm_head, shift, and loss are all computed on packed tensors.
            packed_logits = self.lm_head(hidden_states).float()  # (total_tokens, vocab)

            if labels is not None:
                batch_size, seq_len = packed_seq_shape
                # Build shift_labels in the full padded space, then index with the same
                # packed_seq_indices used for hidden_states.
                shift_labels_padded = F.pad(
                    labels[:, 1:].contiguous(), (0, 1), value=-100
                )  # (batch, seq)
                packed_labels = shift_labels_padded.view(batch_size * seq_len)[
                    packed_seq_indices
                ]  # (total_tokens,)

                if shard_config.enable_tensor_parallelism and shard_config.parallel_output:
                    loss = cross_entropy_1d(
                        packed_logits,
                        packed_labels,
                        process_group=shard_config.tensor_parallel_process_group,
                        vocab_size=self.lm_head.out_features,
                        dtype=self.model.dtype,
                        mode="sum",
                    )
                    num_nonzero = (packed_labels != -100).sum()
                    loss = (loss / num_nonzero).squeeze()
                else:
                    from torch.nn import CrossEntropyLoss

                    loss = CrossEntropyLoss(ignore_index=-100)(
                        packed_logits, packed_labels
                    )

            logits = packed_logits
        else:
            # Standard padded-batch path.
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

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


class LlamaAttentionForwards:
    @staticmethod
    def forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        A replacement for the forward method of the `LlamaAttention` class.
        This adds preprocessing of context parallelism that the original transformers
        doesn't have.
        """
        # Packed sequence path: hidden_states is (total_tokens, hidden_dim), position_embeddings
        # are (total_tokens, head_dim) each.  cu_seq_lens_q is injected by llama_model_forward.
        if kwargs.get("cu_seq_lens_q") is not None and past_key_value is None:
            cu_seqlens: torch.Tensor = kwargs["cu_seq_lens_q"]
            max_seqlen: int = kwargs["max_length_q"]

            q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
            k = self.k_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)
            v = self.v_proj(hidden_states).view(-1, self.num_key_value_heads, self.head_dim)

            # cos/sin: (total_tokens, head_dim); unsqueeze_dim=1 adds a broadcast dim over heads
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                softmax_scale=self.scaling,
                causal=self.is_causal,
            )  # (total_tokens, num_heads, head_dim)

            attn_output = attn_output.reshape(-1, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None

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
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # sp: all-to-all communication when introducing ulysses context parallelism
        if sp_mode == "all_to_all":
            attn_output = all_to_all_comm(
                attn_output, sp_group, scatter_dim=1, gather_dim=2
            )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
