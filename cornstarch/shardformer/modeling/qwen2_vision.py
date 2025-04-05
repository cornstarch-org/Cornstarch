import functools
from typing import Callable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    VisionAttention,
    apply_rotary_pos_emb_vision,
    logger,
)

from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_flash_attention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
    ContextParallelDistributionMode,
)

_SUPPORTED_CP_MODE = ["ring_attn"]


class Qwen2VisionModelForwards:
    @staticmethod
    def qwen2_vision_transformer_forward(
        self: Qwen2VisionTransformerPretrainedModel,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        shard_config: ShardConfig = None,
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

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is not None:
                hidden_states = pixel_values
            elif pixel_values_videos is not None:
                hidden_states = pixel_values_videos

        if image_grid_thw is not None:
            grid_thw = image_grid_thw
        elif video_grid_thw is not None:
            grid_thw = video_grid_thw

        assert hidden_states is not None and grid_thw is not None

        if hidden_states.ndim == 3:
            # Slice-based microbatching leaves one more dimension
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        if grid_thw.ndim == 3:
            grid_thw = grid_thw.view(-1, grid_thw.size(-1))

        if stage_manager is None or stage_manager.is_first_stage():
            hidden_states = self.patch_embed(hidden_states)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        sp_mode = shard_config.sequence_parallelism_mode
        sp_group = shard_config.sequence_parallel_process_group
        sp_size = shard_config.sequence_parallel_size
        sp_rank = dist.get_rank(sp_group)

        # Support SP + PP. Later stages have already received the split input.
        if sp_mode == "ring_attn":
            split_input = stage_manager is None or stage_manager.is_first_stage()
            if split_input:
                # FIXME: Qwen2Vision does not have a batched input,
                # but they are concatenated. Current implementation
                # does not work even though somehow it passes the tests.
                # Need to support varlen input.
                ContextParallelBatchSplitUtils.create_context_parallel_split(
                    # fake attention mask
                    torch.empty((1, hidden_states.shape[0]), device="meta"),
                    sp_group,
                    dist_mode=ContextParallelDistributionMode.UNIFORM,
                )

                hidden_states = ContextParallelBatchSplitUtils.split_batch(
                    hidden_states,
                    sp_group,
                )

            # Recompute cu_seqlens and rotary_pos_emb here
            # FIXME: this doesn't work for arbitrary length
            # split should be done in image-wise
            cu_seqlens_chunks = cu_seqlens[1:].chunk(sp_size, dim=0)
            cu_seqlens = cu_seqlens_chunks[sp_rank]
            if sp_rank > 0:
                cu_seqlens -= cu_seqlens_chunks[sp_rank - 1][-1]
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            grid_thw = grid_thw.chunk(sp_size, dim=0)[sp_rank]
            rotary_pos_emb = self.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.blocks))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.blocks))

        for blk in self.blocks[start_idx:end_idx]:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, None, position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        ContextParallelBatchSplitUtils.clear_cache()

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return {"hidden_states": hidden_states}

        return self.merger(hidden_states)


class Qwen2VisionAttentionForwards:
    # This replaces VisionAttention, not Qwen2VLAttention.
    # Qwen2VLAttention is for LLM.
    @staticmethod
    def forward(
        self: VisionAttention,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        shard_config: Optional[ShardConfig] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        seq_length = hidden_states.shape[0]

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

        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # FIXME: to avoid using varlen function
        # which context parallelism doesn't support,
        # Cornstarch forces to use the same length for all sequences
        # in the same batch.
        if len(cu_seqlens) == 1:
            # shape: (batch, seq_len, num_heads, head_dim)
            q, k, v = (
                t.view(1, cu_seqlens.item(), self.num_heads, -1).transpose(1, 2)
                for t in [q, k, v]
            )
        elif len(cu_seqlens) > 1:
            lengths = torch.diff(cu_seqlens).tolist()
            assert all(
                l == lengths[0] for l in lengths
            ), "All sequences in the same batch must have the same length."

            # shape: (batch, seq_len, num_heads, head_dim)
            q, k, v = (
                t.view(len(lengths), lengths[0], self.num_heads, -1).transpose(1, 2)
                for t in [q, k, v]
            )
        else:
            raise ValueError("cu_seqlens must have at least one element")

        if sp_mode == "ring_attn":
            attention_interface: Callable = functools.partial(
                context_parallel_flash_attention, sp_group=sp_group
            )
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

        attn_output, attn_weights = attention_interface(
            self,
            q,
            k,
            v,
            attention_mask=None,
        )

        attn_output = attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()

        attn_output = self.proj(attn_output)
        return attn_output
