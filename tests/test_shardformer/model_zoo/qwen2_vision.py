import torch
import torch.distributed as dist
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLVisionConfig,
)

from cornstarch.shardformer.layers.operation import gather_forward_split_backward

from ..utils import ModelClassBase


class Qwen2VisionTransformerBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2VisionTransformerPretrainedModel,
            Qwen2VLVisionConfig(
                hidden_size=256,
                embed_dim=64,
                num_heads=8,
                num_hidden_layers=4,
                depth=4,
                use_cache=False,
            ),
        )
        self.col_layers_to_check = [
            "blocks[0].attn.proj",
            "blocks[0].mlp.fc2",
        ]
        self.row_layers_to_check = [
            "blocks[0].attn.qkv",
            "blocks[0].mlp.fc1",
        ]
        self.norm_layers_to_check = [
            "blocks[0].norm1",
            "blocks[0].norm2",
        ]

    def loss_fn(
        self, x: torch.Tensor, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        sp_size = dist.get_world_size(sp_group)
        if sp_group is not None and sp_size > 1:
            gathered_states = gather_forward_split_backward(
                x, dim=0, process_group=sp_group, grad_scale=sp_size
            )
            output = gathered_states.mean()
        else:
            output = x.mean()

        return output

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = 256  # minimum pixel size
        num_grid = image_size // self.config.patch_size
        num_channels = self.config.in_channels

        pixel_values = torch.randn(
            num_grid**2,
            num_channels * self.config.temporal_patch_size * self.config.patch_size**2,
        )
        image_grid_thw = torch.tensor([[1, num_grid, num_grid]])

        # Stacking is required to run models with get_micro_batch()
        # as Qwen2VisionImageProcessor flattens all images.
        return {
            "pixel_values": torch.stack([pixel_values] * num_batch, dim=0),
            "image_grid_thw": torch.stack([image_grid_thw] * num_batch, dim=0),
        }

    @property
    def num_tokens(self) -> int:
        image_size = 256  # minimum pixel size
        num_grid = image_size // self.config.patch_size

        return num_grid**2 // self.config.spatial_merge_size**2


class Qwen2VisionTransformerVarlenBase(Qwen2VisionTransformerBase):
    """Qwen2Vision model with variable-size images for varlen SP testing.

    Each image in the batch can have a different (h, w) grid size.
    The only constraint is that h and w are divisible by spatial_merge_size=2,
    which is required by rot_pos_emb's reshape.
    """

    def data_gen_fn(self, num_batch: int) -> dict:
        # Grid sizes that are all divisible by spatial_merge_size=2.
        grid_sizes = [(4, 4), (8, 8), (4, 8), (6, 6)]
        patch_feature_dim = (
            self.config.in_channels
            * self.config.temporal_patch_size
            * self.config.patch_size**2
        )

        pixel_values_list = []
        grid_thw_list = []
        for b in range(num_batch):
            h, w = grid_sizes[b % len(grid_sizes)]
            pixel_values_list.append(
                torch.randn(h * w, patch_feature_dim)
            )
            grid_thw_list.append([1, h, w])

        return {
            "pixel_values": torch.cat(pixel_values_list, dim=0),  # [total_T, H]
            "image_grid_thw": torch.tensor(grid_thw_list),  # [num_batch, 3]
        }

    # loss_fn is inherited from Qwen2VisionTransformerBase unchanged:
    # the encoder output is SP-split (gather→merge→slice in model forward),
    # so the existing gather-then-mean in the base class is correct.
