import torch
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.vit import ViTConfig, ViTModel

from cornstarch.shardformer.layers.operation import gather_forward_split_backward

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
                image_size=224,
                patch_size=14,
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

    def loss_fn(
        self, x: BaseModelOutputWithPooling, sp_group: dist.ProcessGroup = None
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
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
