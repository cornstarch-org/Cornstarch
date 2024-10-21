import torch
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from ..utils import ModelClassBase


class WhisperEncoderBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig(
                d_model=384,
                encoder_ffn_dim=128,
                encoder_attention_heads=16,
                encoder_layers=4,
                is_encoder_decoder=False,
                use_cache=False,
            ),
        )
        self.row_layers_to_check = ["layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["layers[0].self_attn.out_proj"]

    def loss_fn(self, x: BaseModelOutput) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self, num_batch: int) -> dict:
        return dict(
            input_features=torch.rand(num_batch, self.config.num_mel_bins, 3000)
        )
