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
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
    _get_unpad_data,
)
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

from cornstarch.shardformer.layers.context_parallel_ring_attention import (
    context_parallel_varlen_ring_attention_forward,
)

_SUPPORTED_CP_MODE = ["all_to_all", "ring_attn"]


def _zigzag_local_indices(
    sp_rank: int, sp_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    """Return flat token indices for this rank under zigzag-2P partitioning.

    The sequence is split into 2*sp_size roughly equal chunks (sizes differ by at
    most 1 when seq_len is not divisible by 2*sp_size).  The first
    `seq_len % (2*sp_size)` chunks receive one extra token.  Rank i receives
    chunk i and chunk 2*sp_size-1-i, concatenated in that order.
    """
    total_chunks = 2 * sp_size
    base = seq_len // total_chunks
    extra = seq_len % total_chunks  # first `extra` chunks have size base+1

    def _chunk_range(c: int) -> tuple[int, int]:
        start = c * base + min(c, extra)
        size = base + (1 if c < extra else 0)
        return start, start + size

    lo_a, hi_a = _chunk_range(sp_rank)
    lo_b, hi_b = _chunk_range(2 * sp_size - 1 - sp_rank)
    return torch.cat(
        [
            torch.arange(lo_a, hi_a, device=device),
            torch.arange(lo_b, hi_b, device=device),
        ]
    )


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

        # ring_attn packing state (populated in the split_input block below)
        ring_local_idx: Optional[torch.Tensor] = None
        packed_seq_indices_b: Optional[torch.Tensor] = None

        if packed_seq_indices is not None and not (
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
                assert self.config._attn_implementation == "flash_attention_2", (
                    "ring_attn context parallelism requires flash_attention_2. "
                    f"Got {self.config._attn_implementation}"
                )
                sp_rank = dist.get_rank(sp_group)
                seq_len = hidden_states.shape[1]
                ring_local_idx = _zigzag_local_indices(
                    sp_rank, sp_size, seq_len, hidden_states.device
                )
                hidden_states = hidden_states[:, ring_local_idx]  # (B, 2*chunk, H)
                position_ids = ring_local_idx.unsqueeze(0)  # (1, 2*chunk)
            elif sp_mode == "all_to_all":
                hidden_states = split_forward_gather_backward(
                    hidden_states, 1, sp_group, 1 / sp_size
                )

            # Recompute position embeddings after split
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Packed sequence path: pack hidden_states once before the layer loop.
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
            cos = cos.expand(batch_size, seq_len, -1).reshape(batch_size * seq_len, -1)[
                packed_seq_indices
            ]
            sin = sin.expand(batch_size, seq_len, -1).reshape(batch_size * seq_len, -1)[
                packed_seq_indices
            ]
            position_embeddings = (cos, sin)
            flash_attn_kwargs["cu_seq_lens_q"] = packed_seq_cu_seqlens
            flash_attn_kwargs["cu_seq_lens_k"] = packed_seq_cu_seqlens
            flash_attn_kwargs["max_length_q"] = packed_seq_max_seqlen
            flash_attn_kwargs["max_length_k"] = packed_seq_max_seqlen
            attn_mask = None
        elif packed_seq_indices is not None:
            # Non-first PP stage: hidden_states and position_embeddings are already packed.
            flash_attn_kwargs["cu_seq_lens_q"] = packed_seq_cu_seqlens
            flash_attn_kwargs["cu_seq_lens_k"] = packed_seq_cu_seqlens
            flash_attn_kwargs["max_length_q"] = packed_seq_max_seqlen
            flash_attn_kwargs["max_length_k"] = packed_seq_max_seqlen
            attn_mask = None

        # ring_attn packing: pack chunk_a and chunk_b SEPARATELY to produce 2B
        # sub-sequences ordered as [chunk_a_s0,...,chunk_a_sB-1,chunk_b_s0,...,chunk_b_sB-1].
        # This gives a consistent causal offset per sub-sequence (chunk_a tokens share lo_a,
        # chunk_b tokens share lo_b) and allows the ring_attn layer to derive the
        # chunk_a / chunk_b boundary from cu_seqlens_q[B].
        ring_attn_kwargs: dict = {}
        if (
            sp_mode == "ring_attn"
            and self.config._attn_implementation == "flash_attention_2"
            and not use_cache
            and attention_mask is not None
            and (stage_manager is None or stage_manager.is_first_stage())
        ):
            assert ring_local_idx is not None
            B, local_len = hidden_states.shape[:2]
            hidden_dim = hidden_states.shape[-1]

            # Compute chunk_a_size / chunk_b_size using the same uneven-split
            # arithmetic as _zigzag_local_indices.
            orig_seq_len = attention_mask.shape[1]
            _total_chunks = 2 * sp_size
            _base = orig_seq_len // _total_chunks
            _extra = orig_seq_len % _total_chunks
            chunk_a_size = _base + (1 if sp_rank < _extra else 0)
            chunk_b_size = local_len - chunk_a_size

            lo_a = int(ring_local_idx[0].item())
            lo_b = int(ring_local_idx[chunk_a_size].item())

            # Separate local hidden_states and masks for each chunk
            chunk_a_hs = hidden_states[:, :chunk_a_size]  # (B, chunk_a_size, H)
            chunk_b_hs = hidden_states[:, chunk_a_size:]  # (B, chunk_b_size, H)
            mask_a = attention_mask[
                :, ring_local_idx[:chunk_a_size]
            ]  # (B, chunk_a_size)
            mask_b = attention_mask[
                :, ring_local_idx[chunk_a_size:]
            ]  # (B, chunk_b_size)

            idx_a, cu_seqlens_a, max_seqlen_a = _get_unpad_data(mask_a)
            idx_b, cu_seqlens_b, max_seqlen_b = _get_unpad_data(mask_b)

            # Pack hidden_states: [chunk_a_tokens | chunk_b_tokens]
            packed_a = index_first_axis(
                chunk_a_hs.reshape(B * chunk_a_size, hidden_dim), idx_a
            )
            packed_b = index_first_axis(
                chunk_b_hs.reshape(B * chunk_b_size, hidden_dim), idx_b
            )
            hidden_states = torch.cat([packed_a, packed_b], dim=0)

            # Pack position embeddings the same way
            cos_full, sin_full = position_embeddings  # (1 or B, local_len, head_dim)
            cos_full = cos_full.expand(B, local_len, -1)
            sin_full = sin_full.expand(B, local_len, -1)
            cos_a = cos_full[:, :chunk_a_size].reshape(B * chunk_a_size, -1)[idx_a]
            cos_b = cos_full[:, chunk_a_size:].reshape(B * chunk_b_size, -1)[idx_b]
            sin_a = sin_full[:, :chunk_a_size].reshape(B * chunk_a_size, -1)[idx_a]
            sin_b = sin_full[:, chunk_a_size:].reshape(B * chunk_b_size, -1)[idx_b]
            position_embeddings = (
                torch.cat([cos_a, cos_b], dim=0),
                torch.cat([sin_a, sin_b], dim=0),
            )

            # cu_seqlens_q: 2B+1 entries
            # [0, valid_a_0, ..., total_a, total_a+valid_b_0, ..., total_a+total_b]
            cu_seqlens_q = torch.cat(
                [cu_seqlens_a, cu_seqlens_a[-1] + cu_seqlens_b[1:]]
            )
            packed_seq_max_seqlen = max(int(max_seqlen_a), int(max_seqlen_b))

            # q_seq_offsets: (2B,) — all chunk_a offsets, then all chunk_b offsets
            valid_lens = attention_mask.sum(dim=1).long()  # (B,)
            q_offsets_a = valid_lens.clamp(max=lo_a).int()  # (B,)
            q_offsets_b = valid_lens.clamp(max=lo_b).int()  # (B,)
            q_seq_offsets = torch.cat([q_offsets_a, q_offsets_b])  # (2B,)

            # cu_seqlens_k_global: 2B+1 entries matching cu_seqlens_q ordering
            # (same-sample K length for both chunk_a and chunk_b sub-sequences)
            k_lens = torch.cat([valid_lens, valid_lens])  # (2B,)
            cu_seqlens_k_global = torch.zeros(
                2 * B + 1, dtype=torch.int32, device=hidden_states.device
            )
            cu_seqlens_k_global[1:] = k_lens.cumsum(0).int()

            # Store for PP propagation (packed_seq_cu_seqlens carries cu_seqlens_q).
            # packed_seq_indices_b carries idx_b so the last PP stage can pack labels.
            packed_seq_indices = idx_a  # placeholder for PP is_first_stage detection
            packed_seq_indices_b = idx_b
            packed_seq_cu_seqlens = cu_seqlens_q
            packed_seq_shape = (B, local_len)

            ring_attn_kwargs["cu_seqlens_q"] = cu_seqlens_q
            ring_attn_kwargs["cu_seqlens_k_global"] = cu_seqlens_k_global
            ring_attn_kwargs["q_seq_offsets"] = q_seq_offsets
            ring_attn_kwargs["max_seqlen_q"] = packed_seq_max_seqlen
            ring_attn_kwargs["max_seqlen_k"] = int(valid_lens.max().item())
            attn_mask = None
        elif sp_mode == "ring_attn" and packed_seq_indices is not None:
            # Non-first PP stage: ring_attn_kwargs were propagated from prior stage.
            ring_attn_kwargs = {
                "cu_seqlens_q": packed_seq_cu_seqlens,
                "cu_seqlens_k_global": packed_seq_shape,  # repurposed field, see below
                "q_seq_offsets": flash_attn_kwargs.pop("q_seq_offsets", None),
                "max_seqlen_q": packed_seq_max_seqlen,
                "max_seqlen_k": flash_attn_kwargs.pop("max_seqlen_k_ring", 0),
            }
            attn_mask = None

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.layers))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.layers))

        kwargs = {}
        kwargs.update(flash_attn_kwargs)
        if ring_attn_kwargs:
            kwargs.update(ring_attn_kwargs)

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
                outputs["packed_seq_indices_b"] = packed_seq_indices_b
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
        packed_seq_indices: Optional[torch.Tensor] = None,
        packed_seq_indices_b: Optional[torch.Tensor] = None,
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
        resolved_use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
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

        # For ring_attn, llama_model_forward packs hidden_states into a 2D tensor
        # (total_tokens, hidden_dim). Pre-compute lm_head / loss metadata here, while
        # attention_mask and the original labels are still available.
        ring_attn_packed = False
        ring_chunk_a_size = ring_chunk_b_size = 0
        ring_idx_a = ring_idx_b = None
        if (
            sp_mode == "ring_attn"
            and self.config._attn_implementation == "flash_attention_2"
            and not resolved_use_cache
            and attention_mask is not None
            and (stage_manager is None or stage_manager.is_first_stage())
        ):
            ring_attn_packed = True
            _sp_group = shard_config.sequence_parallel_process_group
            _sp_rank = dist.get_rank(_sp_group)
            _sp_size = shard_config.sequence_parallel_size
            _orig_sl = attention_mask.shape[1]
            _local_idx = _zigzag_local_indices(
                _sp_rank, _sp_size, _orig_sl, attention_mask.device
            )
            _total_c = 2 * _sp_size
            ring_chunk_a_size = (_orig_sl // _total_c) + (
                1 if _sp_rank < (_orig_sl % _total_c) else 0
            )
            ring_chunk_b_size = int(_local_idx.shape[0]) - ring_chunk_a_size
            _mask_a = attention_mask[:, _local_idx[:ring_chunk_a_size]]
            _mask_b = attention_mask[:, _local_idx[ring_chunk_a_size:]]
            ring_idx_a, _, _ = _get_unpad_data(_mask_a)
            ring_idx_b, _, _ = _get_unpad_data(_mask_b)

        if (
            sp_mode == "ring_attn"
            and shard_config.parallel_output
            and (stage_manager is None or stage_manager.is_first_stage())
        ):
            # Split labels with zigzag partitioning, same as hidden_states.
            # Only the first stage has the original labels; other stages receive
            # hidden_states as input and do not process labels here.
            sp_group = shard_config.sequence_parallel_process_group
            sp_rank = dist.get_rank(sp_group)
            sp_size = shard_config.sequence_parallel_size
            local_idx = _zigzag_local_indices(
                sp_rank, sp_size, labels.shape[1], labels.device
            )
            labels = labels[:, local_idx]

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
            packed_seq_indices_b = outputs.get("packed_seq_indices_b", packed_seq_indices_b)
            packed_seq_shape = outputs.get("packed_seq_shape", packed_seq_shape)

        hidden_states = outputs[0]

        loss = None
        if ring_attn_packed:
            # Ring-attention packed path: hidden_states is (total_tokens, hidden_dim)
            # where tokens come from chunk_a (0..valid_a) then chunk_b (valid_a..end).
            packed_logits = self.lm_head(hidden_states).float()

            if labels is not None:
                # labels is (B, local_len) — already zigzag-cut when parallel_output=True.
                # Shift within each local chunk; the last position of each chunk gets -100
                # because the true next-token label lies on another rank / out of range.
                B = attention_mask.shape[0]
                shift_a = F.pad(
                    labels[:, 1:ring_chunk_a_size].contiguous(), (0, 1), value=-100
                )  # (B, ring_chunk_a_size)
                shift_b = F.pad(
                    labels[:, ring_chunk_a_size + 1 :].contiguous(), (0, 1), value=-100
                )  # (B, ring_chunk_b_size)
                packed_labels_a = shift_a.reshape(B * ring_chunk_a_size)[ring_idx_a]
                packed_labels_b = shift_b.reshape(B * ring_chunk_b_size)[ring_idx_b]
                packed_labels = torch.cat([packed_labels_a, packed_labels_b])

                if (
                    shard_config.enable_tensor_parallelism
                    and shard_config.parallel_output
                ):
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
        elif sp_mode == "ring_attn" and packed_seq_indices is not None:
            # Last PP stage with ring_attn: the first stage already partitioned the
            # sequence, but ring_attn_packed was not set here.  Re-derive the zigzag
            # indices from labels (which arrive as the original full-length tensor on
            # every PP stage) and pack labels the same way the first stage does.
            packed_logits = self.lm_head(hidden_states).float()
            if labels is not None:
                _sp_group = shard_config.sequence_parallel_process_group
                _sp_rank = dist.get_rank(_sp_group)
                _sp_size = shard_config.sequence_parallel_size
                _orig_sl = labels.shape[1]
                _local_idx = _zigzag_local_indices(
                    _sp_rank, _sp_size, _orig_sl, labels.device
                )
                _total_c = 2 * _sp_size
                _chunk_a = (_orig_sl // _total_c) + (
                    1 if _sp_rank < (_orig_sl % _total_c) else 0
                )
                B, local_len = packed_seq_shape
                _chunk_b = local_len - _chunk_a
                local_labels = labels[:, _local_idx]  # (B, local_len)
                shift_a = F.pad(
                    local_labels[:, 1:_chunk_a].contiguous(), (0, 1), value=-100
                )  # (B, _chunk_a)
                shift_b = F.pad(
                    local_labels[:, _chunk_a + 1 :].contiguous(), (0, 1), value=-100
                )  # (B, _chunk_b)
                packed_labels_a = shift_a.reshape(B * _chunk_a)[packed_seq_indices]
                packed_labels_b = shift_b.reshape(B * _chunk_b)[packed_seq_indices_b]
                packed_labels = torch.cat([packed_labels_a, packed_labels_b])

                if (
                    shard_config.enable_tensor_parallelism
                    and shard_config.parallel_output
                ):
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
        elif packed_seq_indices is not None:
            # Non-SP packed sequence path: hidden_states is (total_tokens, hidden_dim).
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

                if (
                    shard_config.enable_tensor_parallelism
                    and shard_config.parallel_output
                ):
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
            k = self.k_proj(hidden_states).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            v = self.v_proj(hidden_states).view(
                -1, self.num_key_value_heads, self.head_dim
            )

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

        # Varlen ring_attn path: hidden_states is packed (2-D: total_tokens × hidden_dim).
        # The standard view+transpose+RoPE path assumes 3-D (B, S, H) input, so we must
        # handle projections + RoPE separately here before entering that path.
        if sp_mode == "ring_attn" and kwargs.get("cu_seqlens_q") is not None:
            assert self.config._attn_implementation == "flash_attention_2", (
                "ring_attn context parallelism requires flash_attention_2. "
                f"Got {self.config._attn_implementation}"
            )
            cu_seqlens_q = kwargs["cu_seqlens_q"]
            cu_seqlens_k_global = kwargs["cu_seqlens_k_global"]
            q_seq_offsets = kwargs["q_seq_offsets"]
            max_seqlen_q = kwargs["max_seqlen_q"]
            max_seqlen_k = kwargs["max_seqlen_k"]

            # hidden_states is (total_tokens, hidden_dim); reshape projections directly.
            q_varlen = query_states.view(-1, self.num_heads, self.head_dim)
            k_varlen = key_states.view(-1, self.num_key_value_heads, self.head_dim)
            v_varlen = value_states.view(-1, self.num_key_value_heads, self.head_dim)

            # cos/sin are packed (total_tokens, head_dim); unsqueeze_dim=1 broadcasts over heads.
            cos, sin = position_embeddings
            q_varlen, k_varlen = apply_rotary_pos_emb(
                q_varlen, k_varlen, cos, sin, unsqueeze_dim=1
            )

            attn_output, _ = context_parallel_varlen_ring_attention_forward(
                self,
                q_varlen,
                k_varlen,
                v_varlen,
                sp_group=sp_group,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_global=cu_seqlens_k_global,
                q_seq_offsets=q_seq_offsets,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            # attn_output: (total_tokens, heads, D) → reshape to (total_tokens, hidden_size)
            attn_output = attn_output.reshape(-1, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None
        elif sp_mode == "ring_attn":
            # ring_attn without packed sequences (e.g. use_cache=True) – unsupported.
            raise NotImplementedError(
                "ring_attn context parallelism requires packed sequences "
                "(flash_attention_2 without use_cache)."
            )

        # Non-ring_attn paths: apply all-to-all (if needed), reshape, RoPE, and KV cache.
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
