import torch
import torch.distributed as dist
from colossalai.shardformer.layer._operation import reduce_forward
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
                image_size=224,
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
            "vision_model.pre_layrnorm",
        ]

    def loss_fn(
        self, x: BaseModelOutputWithPooling, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        sp_size = dist.get_world_size(sp_group)
        if sp_group is not None and sp_size > 1:
            output = (
                reduce_forward(x.pooler_output, sp_group, grad_scale=sp_size) / sp_size
            ).mean()
        else:
            output = x.pooler_output.mean()

        return output

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }
