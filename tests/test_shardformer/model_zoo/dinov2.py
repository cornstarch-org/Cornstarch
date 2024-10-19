import copy

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.dinov2 import Dinov2Config, Dinov2Model

from ..utils import ModelClassBase


class Dinov2ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config(
                hidden_size=256,
                intermediate_size=256,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
            ),
        )
        self.col_layers_to_check = ["encoder.layer[0].attention.output.dense"]
        self.row_layers_to_check = ["encoder.layer[0].attention.attention.query"]

    def loss_fn(self, x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    # HF does not provide Dinov2 flash attention yet.
    # Use SDPA implementation and compare against ColoAttention.
    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "sdpa"
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
