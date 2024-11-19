import copy

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vit import ViTConfig, ViTModel

from ..utils import ModelClassBase


class ViTModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            ViTModel,
            ViTConfig(
                hidden_size=256,
                intermediate_size=64,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
            ),
        )
        self.row_layers_to_check = [
            "encoder.layer[0].attention.attention.query",
            "encoder.layer[0].attention.attention.key",
            # "encoder.layer[0].intermediate.dense",
        ]
        self.col_layers_to_check = [
            "encoder.layer[0].attention.output.dense",
            # "encoder.layer[0].output.dense",
        ]
        self.norm_layers_to_check = [
            "encoder.layer[0].layernorm_before",
            "encoder.layer[0].layernorm_after",
        ]

    def loss_fn(self, x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    # ViT does not support FlashAttention.
    # Use eager implementation
    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "eager"
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
