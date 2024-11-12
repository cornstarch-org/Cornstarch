import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from ..utils import ModelClassBase

qwen2_config = Qwen2Config(
    hidden_size=512,
    intermediate_size=64,
    num_attention_heads=16,
    num_key_value_heads=8,
    num_hidden_layers=4,
    use_cache=False,
)


class Qwen2ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen2Model, qwen2_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["layers[0].self_attn.o_proj"]

    def loss_fn(self, x: BaseModelOutputWithPast) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
        }

        return input


class Qwen2ForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen2ForCausalLM, qwen2_config)
        # embed_tokens have different dimension, so skip checking embed_tokens here
        self.row_layers_to_check = ["model.layers[0].self_attn.q_proj"]
        self.col_layers_to_check = ["model.layers[0].self_attn.o_proj"]

    def loss_fn(self, x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
        }
        input["labels"] = input["input_ids"]
        return input
