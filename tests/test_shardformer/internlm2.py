import copy

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.internlm2 import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2Model,
)

from .utils import ModelClassBase

internlm2_config = InternLM2Config(
    hidden_size=256,
    intermediate_size=256,
    num_attention_heads=16,
    num_key_value_heads=8,
    num_hidden_layers=4,
    use_cache=False,
)


class InternLM2ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(InternLM2Model, internlm2_config)
        self.row_layers_to_check = ["layers[0].attention.wqkv", "tok_embeddings"]
        self.col_layers_to_check = ["layers[0].attention.wo"]

    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "flash_attention_2" if fa else "eager"
        config.attn_implementation = config._attn_implementation
        return self.model_class(config)

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


class InternLM2ForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(InternLM2ForCausalLM, internlm2_config)
        self.row_layers_to_check = [
            "model.layers[0].attention.wqkv",
            "model.tok_embeddings",
        ]
        self.col_layers_to_check = ["model.layers[0].attention.wo"]

    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "flash_attention_2" if fa else "eager"
        config.attn_implementation = config._attn_implementation
        return self.model_class(config)

    def loss_fn(self, x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
        }
        input["labels"] = input["input_ids"]

        return input
