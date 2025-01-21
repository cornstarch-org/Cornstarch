import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLVisionConfig,
)

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

    def loss_fn(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = 256  # minimum pixel size
        num_grid = image_size // self.config.patch_size
        num_channels = self.config.in_channels
        return [
            # hidden_states
            torch.randn(
                num_batch,
                num_grid**2,
                num_channels
                * self.config.temporal_patch_size
                * self.config.patch_size**2,
            ),
            # grid_thw
            torch.tensor([[1, num_grid, num_grid]] * num_batch),
        ]
