import torch
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioEncoderConfig,
)

from ..utils import ModelClassBase


class Qwen2AudioEncoderBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2AudioEncoder,
            Qwen2AudioEncoderConfig(
                max_source_positions=64,
                d_model=64,
                encoder_ffn_dim=64,
                encoder_attention_heads=16,
                encoder_layers=4,
                is_encoder_decoder=False,
                use_cache=False,
            ),
        )

        self.col_layers_to_check = ["layers[0].self_attn.out_proj"]
        self.row_layers_to_check = ["layers[0].self_attn.q_proj"]

    def loss_fn(
        self, x: BaseModelOutput, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self, num_batch: int) -> dict:
        return dict(input_features=torch.rand(num_batch, self.config.num_mel_bins, 128))
