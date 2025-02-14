import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel

from ..utils import ModelClassBase


class CLIPModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            CLIPVisionModel,
            CLIPVisionConfig(
                hidden_size=256,
                intermediate_size=256,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
            ),
        )
        self.col_layers_to_check = [
            "vision_model.encoder.layers[0].self_attn.out_proj",
            "vision_model.encoder.layers[0].mlp.fc2",
        ]
        self.row_layers_to_check = [
            "vision_model.encoder.layers[0].self_attn.q_proj",
            "vision_model.encoder.layers[0].mlp.fc1",
        ]
        self.norm_layers_to_check = [
            "vision_model.encoder.layers[0].layer_norm1",
            "vision_model.encoder.layers[0].layer_norm2",
            "vision_model.pre_layrnorm",
        ]

    def loss_fn(self, x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
