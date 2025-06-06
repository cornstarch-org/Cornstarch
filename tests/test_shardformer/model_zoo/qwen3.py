import torch
import torch.distributed as dist
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from ..utils import ModelClassBase

qwen3_config = Qwen3Config(
    hidden_size=256,
    intermediate_size=64,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=4,
    use_cache=False,
)


class Qwen3ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen3Model, qwen3_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["layers[0].self_attn.o_proj"]

    def loss_fn(
        self, x: BaseModelOutputWithPast, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }

        return input


class Qwen3ForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen3ForCausalLM, qwen3_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["model.layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["model.layers[0].self_attn.o_proj"]

    def loss_fn(
        self, x: CausalLMOutputWithPast, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return x.loss

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }
        input["labels"] = input["input_ids"]
        return input
