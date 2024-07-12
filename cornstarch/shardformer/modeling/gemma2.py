from typing import List, Optional, Tuple, Union

import torch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.layer import AttnMaskType, ColoAttention, cross_entropy_1d
from colossalai.shardformer.shard.shard_config import ShardConfig
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Attention,
    Gemma2ForCausalLM,
    Gemma2Model,
    Gemma2SdpaAttention,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)


class Gemma2PipelineForwards:
    @staticmethod
    def gemma2_model_forward(
        self: Gemma2Model,
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

            # normalized
            # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
            # See https://github.com/huggingface/transformers/pull/29402
            normalizer = torch.tensor(
                self.config.hidden_size**0.5, dtype=hidden_states.dtype
            )
            hidden_states = hidden_states * normalizer

        if cache_position is None:
            cache_position = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            hidden_states,
            cache_position,
            past_key_values,
            output_attentions,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

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
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if stage_manager.is_last_stage():
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if stage_manager.is_last_stage():
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
        else:
            return {"hidden_states": hidden_states}

    @staticmethod
    def gemma2_for_causal_lm_forward(
        self: Gemma2ForCausalLM,
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
        stage_manager: Optional[PipelineStageManager] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        stage_index: Optional[List[int]] = None,
        shard_config: ShardConfig = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.training and (
            self.config._attn_implementation != "eager"
            or shard_config.enable_flash_attention
        ):
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )

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
        outputs = Gemma2PipelineForwards.gemma2_model_forward(
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
            stage_manager=stage_manager,
            hidden_states=hidden_states,
            stage_index=stage_index,
            shard_config=shard_config,
        )

        if stage_manager.is_last_stage():
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            if self.config.final_logit_softcapping is not None:
                logits = logits / self.config.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * self.config.final_logit_softcapping

            logits = logits.float()
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
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
                    loss_fct = CrossEntropyLoss()
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

        hidden_states = outputs.get("hidden_states")
        return {"hidden_states": hidden_states}


class Gemma2Forwards:
    # def gemma2_model_forward_for_flash_attention(
    #     self: Gemma2Model,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     shard_config: Optional[ShardConfig] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:
    #     output_attentions = (
    #         output_attentions
    #         if output_attentions is not None
    #         else self.config.output_attentions
    #     )
    #     output_hidden_states = (
    #         output_hidden_states
    #         if output_hidden_states is not None
    #         else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = (
    #         return_dict if return_dict is not None else self.config.use_return_dict
    #     )

    #     if (input_ids is None) ^ (inputs_embeds is not None):
    #         raise ValueError(
    #             "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
    #         )

    #     if self.gradient_checkpointing and self.training and use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
    #         )
    #         use_cache = False

    #     batch_size, seq_length = (
    #         input_ids.shape if input_ids is not None else inputs_embeds.shape
    #     )

    #     if inputs_embeds is None:
    #         inputs_embeds = self.embed_tokens(input_ids)

    #     if cache_position is None:
    #         cache_position = torch.arange(
    #             0, inputs_embeds.shape[1], device=inputs_embeds.device
    #         )

    #     if position_ids is None:
    #         position_ids = cache_position.unsqueeze(0)

    #     causal_mask = self._update_causal_mask(
    #         attention_mask,
    #         inputs_embeds,
    #         cache_position,
    #         past_key_values,
    #         output_attentions,
    #     )

    #     # embed positions
    #     hidden_states = inputs_embeds

    #     # normalized
    #     # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
    #     # See https://github.com/huggingface/transformers/pull/29402
    #     normalizer = torch.tensor(
    #         self.config.hidden_size**0.5, dtype=hidden_states.dtype
    #     )
    #     hidden_states = hidden_states * normalizer

    #     all_hidden_states = () if output_hidden_states else None
    #     all_self_attns = () if output_attentions else None

    #     for decoder_layer in self.layers:
    #         if output_hidden_states:
    #             all_hidden_states += (hidden_states,)

    #         if self.gradient_checkpointing and self.training:
    #             layer_outputs = self._gradient_checkpointing_func(
    #                 decoder_layer.__call__,
    #                 hidden_states,
    #                 causal_mask,
    #                 position_ids,
    #                 past_key_values,
    #                 output_attentions,
    #                 use_cache,
    #                 cache_position,
    #             )
    #         else:
    #             layer_outputs = decoder_layer(
    #                 hidden_states,
    #                 attention_mask=causal_mask,
    #                 position_ids=position_ids,
    #                 past_key_value=past_key_values,
    #                 output_attentions=output_attentions,
    #                 use_cache=use_cache,
    #                 cache_position=cache_position,
    #             )

    #         hidden_states = layer_outputs[0]

    #         if output_attentions:
    #             all_self_attns += (layer_outputs[1],)

    #     hidden_states = self.norm(hidden_states)

    #     # add hidden states from the last decoder layer
    #     if output_hidden_states:
    #         all_hidden_states += (hidden_states,)

    #     next_cache = past_key_values if use_cache else None

    #     if not return_dict:
    #         return tuple(
    #             v
    #             for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
    #             if v is not None
    #         )
    #     return BaseModelOutputWithPast(
    #         last_hidden_state=hidden_states,
    #         past_key_values=next_cache,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attns,
    #     )

    # def gemma2_flash_attention_forward(
    #     self: Union[Gemma2Attention, Gemma2SdpaAttention],
    #     hidden_states: torch.Tensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_value: Optional[Cache] = None,
    #     output_attentions: bool = False,
    #     use_cache: bool = False,
    #     cache_position: Optional[torch.LongTensor] = None,
    # ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    #     bsz, q_len, _ = hidden_states.size()

    #     query_states = self.q_proj(hidden_states)
    #     key_states = self.k_proj(hidden_states)
    #     value_states = self.v_proj(hidden_states)

    #     query_states = query_states.view(
    #         bsz, q_len, self.num_heads, self.head_dim
    #     ).transpose(1, 2)
    #     key_states = key_states.view(
    #         bsz, q_len, self.num_key_value_heads, self.head_dim
    #     ).transpose(1, 2)
    #     value_states = value_states.view(
    #         bsz, q_len, self.num_key_value_heads, self.head_dim
    #     ).transpose(1, 2)

    #     cos, sin = self.rotary_emb(value_states, position_ids)
    #     query_states, key_states = apply_rotary_pos_emb(
    #         query_states, key_states, cos, sin
    #     )

    #     if past_key_value is not None:
    #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #         cache_kwargs = {
    #             "sin": sin,
    #             "cos": cos,
    #             "sliding_window": self.sliding_window,
    #             "cache_position": cache_position,
    #         }
    #         key_states, value_states = past_key_value.update(
    #             key_states, value_states, self.layer_idx, cache_kwargs
    #         )

    #     key_states = repeat_kv(key_states, self.num_key_value_groups)
    #     value_states = repeat_kv(value_states, self.num_key_value_groups)

    #     attn_output = ColoAttention.attention(
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         attention_mask_type=AttnMaskType.CAUSAL,
    #     )

    #     attn_output = attn_output.transpose(1, 2).contiguous()
    #     attn_output = attn_output.reshape(bsz, q_len, -1)

    #     attn_output = self.o_proj(attn_output)

    #     return attn_output, None, past_key_value

    def gemma2_for_causal_lm_forward_with_dist_cross_entropy(
        self: Gemma2ForCausalLM,
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
        shard_config: Optional[ShardConfig] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.training and (
            self.config._attn_implementation != "eager"
            or shard_config.enable_flash_attention
        ):
            logger.warning_once(
                "It is strongly recommended to train Gemma2 models with the `eager` attention implementation "
                f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
            )
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
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            new_vocab_size = logits.shape[-1]
            shift_logits = shift_logits.view(-1, new_vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
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
