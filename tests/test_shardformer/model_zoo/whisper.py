import torch
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder
from cornstarch.shardformer.layers.operation import gather_forward_split_backward

from ..utils import ModelClassBase


class WhisperEncoderBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig(
                max_source_positions=256,
                max_target_positions=64,
                d_model=64,
                encoder_ffn_dim=64,
                encoder_attention_heads=8,
                encoder_layers=4,
                is_encoder_decoder=False,
                use_cache=False,
            ),
        )
        self.row_layers_to_check = ["layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["layers[0].self_attn.out_proj"]

    def loss_fn(
        self, x: BaseModelOutput, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        sp_size = dist.get_world_size(sp_group)
        if sp_group is not None and sp_size > 1:
            gathered_states = gather_forward_split_backward(
                x.last_hidden_state, dim=1, process_group=sp_group, grad_scale=sp_size
            )
            output = gathered_states.mean()
        else:
            output = x.last_hidden_state.mean()

        return output

    def data_gen_fn(self, num_batch: int) -> dict:
        return dict(input_features=torch.rand(num_batch, self.config.num_mel_bins, 512))
