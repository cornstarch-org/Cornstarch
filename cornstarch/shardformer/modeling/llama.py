from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import ColoAttention, cross_entropy_1d
from colossalai.shardformer.shard.shard_config import ShardConfig
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaModel,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)


class LlamaPipelineForwards:
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
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: Optional[ShardConfig] = None,
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if stage_manager.is_first_stage():
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

        past_seen_tokens = 0
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            )

        if stage_manager.is_first_stage():
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

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
        if use_cache:
            logger.warning_once(
                "use_cache=True is not supported for pipeline models at the moment."
            )
            use_cache = False

        if shard_config.enable_flash_attention:
            batch_size, seq_length = hidden_states.shape[:2]
            seq_length_with_past = seq_length + past_seen_tokens
            mask_shape = (batch_size, 1, seq_length_with_past, seq_length_with_past)

            causal_mask = ColoAttention.prepare_attn_kwargs(
                mask_shape,
                hidden_states.dtype,
                hidden_states.device,
                q_padding_mask=attention_mask,
                is_causal=True,
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                output_attentions,
            )

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        start_idx, end_idx = stage_index[0], stage_index[1]
        for decoder_layer in self.layers[start_idx:end_idx]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if stage_manager.is_last_stage():
            next_cache = next_decoder_cache if use_cache else None
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        # always return dict for imediate stage
        return {
            "hidden_states": hidden_states,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
        }

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
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

        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = LlamaPipelineForwards.llama_model_forward(
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
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )
        past_key_values = None

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                if (
                    shard_config.enable_tensor_parallelism
                    and shard_config.parallel_output
                ):
                    new_vocab_size = logits.shape[-1]
                    shift_logits = shift_logits.view(-1, new_vocab_size)
                    loss = cross_entropy_1d(
                        shift_logits,
                        shift_labels,
                        process_group=shard_config.tensor_parallel_process_group,
                        vocab_size=self.lm_head.out_features,
                        dtype=self.model.dtype,
                    )
                else:
                    shift_logits = shift_logits.view(-1, self.config.vocab_size)
                    loss = loss_fct(shift_logits, shift_labels)

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

        return {
            "hidden_states": outputs.get("hidden_states"),
            "position_ids": outputs.get("position_ids"),
            "position_embeddings": outputs.get("position_embeddings"),
        }

    def llama_for_sequence_classification_forward(
        self: LlamaForSequenceClassification,
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # TODO(jianghai): left the recording kv-value tensors as () or None type, this feature may be added in the future.
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

        outputs = LlamaPipelineForwards.llama_model_forward(
            self.model,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            batch_size = hidden_states.shape[0]

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.score(hidden_states)

            if self.config.pad_token_id is None and batch_size != 1:
                raise ValueError(
                    "Cannot handle batch sizes > 1 if no padding token is defined."
                )
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (
                        torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
                    ).to(logits.device)
                else:
                    sequence_lengths = -1

            pooled_logits = logits[
                torch.arange(batch_size, device=logits.device), sequence_lengths
            ]

            loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (
                        labels.dtype == torch.long or labels.dtype == torch.int
                    ):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(pooled_logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        pooled_logits.view(-1, self.num_labels), labels.view(-1)
                    )
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(pooled_logits, labels)

            if not return_dict:
                output = (pooled_logits,) + outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return {
            "hidden_states": outputs.get("hidden_states"),
            "position_ids": outputs.get("position_ids"),
            "position_embeddings": outputs.get("position_embeddings"),
        }


class LlamaForwards:
    @staticmethod
    def llama_model_forward_for_flash_attention(
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
        shard_config: Optional[ShardConfig] = None,
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if shard_config.enable_flash_attention:
            batch_size, seq_length = inputs_embeds.shape[:2]
            seq_length_with_past = seq_length + past_seen_tokens
            mask_shape = (batch_size, 1, seq_length_with_past, seq_length_with_past)
            causal_mask = ColoAttention.prepare_attn_kwargs(
                mask_shape,
                inputs_embeds.dtype,
                inputs_embeds.device,
                q_padding_mask=attention_mask,
                is_causal=True,
            )
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                output_attentions,
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def llama_flash_attention_forward(
        self: Union[LlamaAttention, LlamaSdpaAttention],
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        shard_config: Optional[ShardConfig] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        assert isinstance(
            attention_mask, dict
        ), "Flash Attention Error: attention_mask should be a dict."
        attn_output = ColoAttention.attention(
            query_states, key_states, value_states, **attention_mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def llama_for_causal_lm_forward_with_dist_cross_entropy(
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

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
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
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            new_vocab_size = logits.shape[-1]
            shift_logits = shift_logits.view(-1, new_vocab_size)
            loss = cross_entropy_1d(
                shift_logits,
                shift_labels,
                process_group=shard_config.tensor_parallel_process_group,
                vocab_size=self.lm_head.out_features,
                dtype=self.model.dtype,
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
