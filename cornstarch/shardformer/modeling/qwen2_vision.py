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

            hidden_states = self.patch_embed(hidden_states)
            rotary_pos_emb = self.rot_pos_emb(grid_thw)

            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        if stage_manager is not None:
            layers_per_stage = stage_manager.distribute_layers(len(self.blocks))
            start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        else:
            start_idx, end_idx = (0, len(self.blocks))

        for blk in self.blocks[start_idx:end_idx]:
            hidden_states = blk(
                hidden_states, rotary_pos_emb=rotary_pos_emb, cu_seqlens=cu_seqlens
            )

        if not (stage_manager is None or stage_manager.is_last_stage()):
            return {
                "hidden_states": hidden_states,
                "rotary_pos_emb": rotary_pos_emb,
                "cu_seqlens": cu_seqlens,
            }

        return self.merger(hidden_states)
