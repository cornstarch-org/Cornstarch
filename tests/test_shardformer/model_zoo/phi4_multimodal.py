import copy

import torch
import torch.distributed as dist
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalAudioConfig,
    Phi4MultimodalAudioModel,
)

from ..utils import ModelClassBase


class Phi4MultimodalAudioModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            Phi4MultimodalAudioModel,
            Phi4MultimodalAudioConfig(
                hidden_size=256,
                intermediate_size=256,
                num_attention_heads=8,
                num_hidden_layers=4,
                use_cache=False,
            ),
        )
        self.col_layers_to_check = [
            "encoders[0].self_attn.o_proj",
            "encoders[0].feed_forward_in.down_proj",
            "encoders[0].feed_forward_out.down_proj",
        ]
        self.row_layers_to_check = [
            "encoders[0].self_attn.q_proj",
            "encoders[0].feed_forward_in.up_proj",
            "encoders[0].feed_forward_out.up_proj",
        ]

    def loss_fn(
        self, x: torch.Tensor, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        output = x.mean()
        return output.mean()

    def data_gen_fn(self, num_batch: int) -> dict:
        return dict(
            hidden_states=torch.rand(num_batch, 498, self.config.input_size),
            mask=torch.ones(num_batch, 498, dtype=torch.bool),
        )
