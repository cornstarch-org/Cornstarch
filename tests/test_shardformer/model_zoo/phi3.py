import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.phi3 import Phi3Config, Phi3ForCausalLM, Phi3Model

from ..utils import ModelClassBase

phi3_config = Phi3Config(
    hidden_size=256,
    intermediate_size=64,
    num_attention_heads=8,
    num_key_value_heads=8,
    num_hidden_layers=4,
    use_cache=False,
)


class Phi3ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(Phi3Model, phi3_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["layers[0].self_attn.qkv_proj"]
        self.col_layers_to_check = ["layers[0].self_attn.o_proj"]

    def loss_fn(self, x: BaseModelOutputWithPast) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }

        return input


class Phi3ForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(Phi3ForCausalLM, phi3_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["model.layers[0].self_attn.qkv_proj"]
        self.col_layers_to_check = ["model.layers[0].self_attn.o_proj"]

    def loss_fn(self, x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }
        input["labels"] = input["input_ids"]
        return input
