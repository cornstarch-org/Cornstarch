from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.shardformer.shard.shard_config import ShardConfig
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    VisionAttention,
    apply_rotary_pos_emb_vision,
    logger,
)

from cornstarch.shardformer.layers.context_parallel_attention import (
    context_parallel_varlen_flash_attention,
)
from cornstarch.shardformer.layers.operation import gather_forward_split_backward

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

        # Compute full rotary position embeddings (from global grid_thw, unchanged).
        # These are sliced per-rank below in the SP path.
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        full_cos, full_sin = emb.cos(), emb.sin()

        # Compute global cu_seqlens (cumulative per-image token counts).
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

        if sp_mode == "ring_attn":
            # Compute per-image chunk assignment for this rank.
            # Rank r takes tokens [n_i*r//p, n_i*(r+1)//p) from image i.
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()  # [N]
            local_seqlens = (
                seqlens * (sp_rank + 1) // sp_size - seqlens * sp_rank // sp_size
            )  # [N]

            # Gather local indices (into the full token sequence) for this rank.
            local_indices = torch.cat(
                [
                    torch.arange(
                        cu_seqlens[i].item() + seqlens[i].item() * sp_rank // sp_size,
                        cu_seqlens[i].item()
                        + seqlens[i].item() * (sp_rank + 1) // sp_size,
                        device=hidden_states.device,
                    )
                    for i in range(len(seqlens))
                ]
            )

            # Local position embeddings and local cu_seqlens_q.
            position_embeddings = (full_cos[local_indices], full_sin[local_indices])
            cu_seqlens_q = F.pad(
                local_seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0
            )

            # Split hidden_states on the first pipeline stage (later stages already
            # receive the split tensor from the previous stage via the pipeline).
            if stage_manager is None or stage_manager.is_first_stage():
                hidden_states = hidden_states[local_indices]

            # Store global cu_seqlens as cu_seqlens_k so the attention forward can
            # build the correct global KV layout after all-gather.
            shard_config._varlen_cu_seqlens_k = cu_seqlens

            block_cu_seqlens = cu_seqlens_q
        else:
            position_embeddings = (full_cos, full_sin)
            block_cu_seqlens = cu_seqlens

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.blocks))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.blocks))

        for blk in self.blocks[start_idx:end_idx]:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    block_cu_seqlens,
                    None,
                    position_embeddings,
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=block_cu_seqlens,
                    position_embeddings=position_embeddings,
                )

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return {"hidden_states": hidden_states}

        # Before the PatchMerger, all-gather the SP-split hidden states so the
        # merger's view(-1, hidden_size * spatial_merge_size**2) groups spatial
        # blocks correctly regardless of image size.
        if sp_mode == "ring_attn" and sp_size > 1:
            hidden_states = gather_forward_split_backward(
                hidden_states, dim=0, process_group=sp_group, grad_scale=1
            )

        merged = self.merger(hidden_states)

        # Re-split after the merger to restore the SP-split state expected by
        # the downstream projector and the multimodal pipeline schedule
        # (encoder_sp_gather=True logic).
        if sp_mode == "ring_attn" and sp_size > 1:
            total = merged.shape[0]
            start = total * sp_rank // sp_size
            end = total * (sp_rank + 1) // sp_size
            merged = merged[start:end]

        return merged


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
        # q, k, v: [seq_length, nheads, head_dim]

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

        if sp_mode == "ring_attn":
            # cu_seqlens holds local chunk boundaries (cu_seqlens_q).
            # cu_seqlens_k is the global per-image boundaries stored by the model forward.
            cu_seqlens_q = cu_seqlens
            cu_seqlens_k = getattr(shard_config, "_varlen_cu_seqlens_k", cu_seqlens)
            max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max())
            max_seqlen_k = int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max())
            attn_output = context_parallel_varlen_flash_attention(
                q,
                k,
                v,
                sp_group,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
            )
        else:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
            attn_output = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=q.shape[-1] ** (-0.5),
                causal=False,
            )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output
