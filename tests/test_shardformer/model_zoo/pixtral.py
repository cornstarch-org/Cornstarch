import copy

import torch
import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

from cornstarch.shardformer.layers.operation import gather_forward_split_backward

from ..utils import ModelClassBase


class PixtralVisionModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(
            PixtralVisionModel,
            PixtralVisionConfig(
                hidden_size=256,
                intermediate_size=256,
                num_attention_heads=8,
                image_size=224,
                num_hidden_layers=4,
                use_cache=False,
            ),
        )
        self.col_layers_to_check = [
            "transformer.layers[0].attention.o_proj",
            "transformer.layers[0].feed_forward.down_proj",
        ]
        self.row_layers_to_check = [
            "transformer.layers[0].attention.q_proj",
            "transformer.layers[0].feed_forward.up_proj",
        ]
        self.norm_layers_to_check = [
            "ln_pre",
            "transformer.layers[0].attention_norm",
            "transformer.layers[0].ffn_norm",
        ]

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

    # PixtralVisionModel does not support FlashAttention.
    # Use eager implementation instead.
    def model_fn(self) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "eager"
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        return {
            "pixel_values": torch.randn(
                num_batch, num_channels, image_size, image_size
            ),
            "image_sizes": torch.tensor(
                [[image_size, image_size] for _ in range(num_batch)]
            ),
        }
