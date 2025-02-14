import copy

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from cornstarch.models.intern_vit import InternVisionConfig, InternVisionModel

from ..utils import ModelClassBase


class InternVisonModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            InternVisionModel,
            InternVisionConfig(
                hidden_size=1024,  # each head has 128 hidden size
                intermediate_size=256,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
                qk_normalization=False,
            ),
        )
        self.row_layers_to_check = [
            "encoder.layers[0].attn.qkv",
            "encoder.layers[0].mlp.fc1",
        ]
        self.col_layers_to_check = [
            "encoder.layers[0].attn.proj",
            "encoder.layers[0].mlp.fc2",
        ]
        self.norm_layers_to_check = [
            "encoder.layers[0].norm1",
            "encoder.layers[0].norm2",
        ]

    def loss_fn(self, x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    # InternVision FlashAttention does not support torch.autocast AMP.
    # Use eager implementation and compare against ColoAttention.
    def model_fn(self) -> InternVisionModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "eager"
        config.use_flash_attn = False
        config.qk_normalization = False
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
