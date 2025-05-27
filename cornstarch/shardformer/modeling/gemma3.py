import functools
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from colossalai.shardformer.layer import dist_cross_entropy
from colossalai.shardformer.layer._operation import (
    all_to_all_comm,
    gather_sp_output,
    split_forward_gather_backward,
)
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.cache_utils import Cache, DynamicCache, HybridCache
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3ForCausalLM,
    Gemma3TextModel,
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


class Gemma3ModelForwards:
    @staticmethod
    def gemma3_model_forward(
        self: Gemma3TextModel,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: Optional[int] = None,
        position_embeddings_global: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_local: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        force_sp_gather: bool = True,  # Set to false only when computing cross entropy
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
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

        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is None or stage_manager.is_first_stage():
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You must specify exactly one of input_ids or inputs_embeds"
                )

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            hidden_states = inputs_embeds

        if use_cache and past_key_values is None and not self.training:
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

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if position_embeddings_global is None:
            # create position embeddings to be shared across the decoder layers
            position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.rotary_emb_local(
                hidden_states, position_ids
            )

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

        if self.config._attn_implementation == "bitfield_attention":
            causal_mask = attention_mask
        else:
            causal_mask = self._update_causal_mask(
                attention_mask,
                hidden_states,
                cache_position,
                past_key_values,
                output_attentions,
            )

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

        kwargs = {}

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[start_idx:end_idx]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    functools.partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    position_embeddings_global,
                    position_embeddings_local,
                    causal_mask,
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
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=causal_mask,
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
                all_self_attns += (layer_outputs[1],)

        BitfieldUtils.clear_cache()
        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            outputs = {
                "hidden_states": hidden_states,
                "cache_position": cache_position,
                "position_embeddings_global": position_embeddings_global,
                "position_embeddings_local": position_embeddings_local,
            }
            if output_hidden_states:
                outputs["all_hidden_states"] = all_hidden_states
            if output_attentions:
                outputs["all_self_attentions"] = all_self_attentions
            outputs.update(kwargs)
            return outputs

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    @staticmethod
    def gemma3_for_causal_lm_forward(
        self: Gemma3ForCausalLM,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        position_embeddings_global: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_local: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        all_hidden_states: Optional[Tuple[torch.FloatTensor]] = (),
        all_self_attentions: Optional[Tuple[torch.Tensor]] = (),
        shard_config: ShardConfig = None,
        offsets_per_rank: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self.training and self.config._attn_implementation != "eager":
            logger.warning_once(
                "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
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

        outputs: BaseModelOutputWithPast = Gemma3ModelForwards.gemma3_model_forward(
            self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            hidden_states=hidden_states,
            all_hidden_states=all_hidden_states,
            all_self_attentions=all_self_attentions,
            shard_config=shard_config,
            force_sp_gather=False,
            offsets_per_rank=offsets_per_rank,
            **kwargs,
        )

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return outputs

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
