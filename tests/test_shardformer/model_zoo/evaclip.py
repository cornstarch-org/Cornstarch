import copy

import torch
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutputWithPooling

from cornstarch.models.evaclip import EvaCLIPVisionConfig, EvaCLIPVisionModel

from ..utils import ModelClassBase


class EvaCLIPModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            EvaCLIPVisionModel,
            EvaCLIPVisionConfig(
                hidden_size=256,
                intermediate_size=256,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
                patch_size=14,
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
        ]

    def loss_fn(
        self, x: BaseModelOutputWithPooling, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return x.pooler_output.mean()

    # HF does not provide EvaCLIP flash attention yet.
    # Use eager implementation and compare against ColoAttention.
    def model_fn(self) -> EvaCLIPVisionModel:
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
