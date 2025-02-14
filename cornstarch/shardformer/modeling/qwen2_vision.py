from typing import Optional

import torch
import torch.nn.functional as F
from colossalai.shardformer.shard.shard_config import ShardConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)


class Qwen2VisionModelForwards:
    @staticmethod
    def qwen2_vision_transformer_forward(
        self: Qwen2VisionTransformerPretrainedModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hidden_states: Optional[torch.FloatTensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        shard_config: ShardConfig = None,
    ) -> torch.Tensor:
        stage_manager = shard_config.pipeline_stage_manager

        if stage_manager is None or stage_manager.is_first_stage():
            if pixel_values is not None:
                hidden_states = pixel_values
                grid_thw = image_grid_thw
            elif pixel_values_videos is not None:
                hidden_states = pixel_values_videos
                grid_thw = video_grid_thw

            if hidden_states.ndim == 3:
                # Slice-based microbatching leaves one more dimension
                hidden_states = hidden_states.view(-1, hidden_states.size(-1))
                grid_thw = grid_thw.view(-1, grid_thw.size(-1))

            hidden_states = self.patch_embed(hidden_states)
            rotary_pos_emb = self.rot_pos_emb(grid_thw)

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

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.blocks))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.blocks))

        for blk in self.blocks[start_idx:end_idx]:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens, rotary_pos_emb
                )
            else:
                hidden_states = blk(
                    hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
                )

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return {
                "hidden_states": hidden_states,
                "rotary_pos_emb": rotary_pos_emb,
                "cu_seqlens": cu_seqlens,
            }

        return self.merger(hidden_states)
