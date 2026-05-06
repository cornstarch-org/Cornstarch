import copy
import torch
import torch.distributed as dist
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama4.modeling_llama4 import (
    Llama4TextConfig,
    Llama4ForCausalLM,
    Llama4TextModel,
)

from ..utils import ModelClassBase

llama_config = Llama4TextConfig(
    hidden_size=256,
    intermediate_size=64,
    intermediate_size_mlp=128,
    attention_chunk_size=64,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=4,
    num_local_experts=2,
    use_cache=False,
)


class Llama4ModelBase(ModelClassBase):
    rtol, atol = 1e-2, 1e-2

    def __init__(self):
        super().__init__(Llama4TextModel, llama_config)
        self.row_layers_to_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
        self.col_layers_to_check = ["layers[0].self_attn.o_proj"]

    def loss_fn(
        self, x: BaseModelOutputWithPast, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    # Llama4 does not support FlashAttention yet.
    def model_fn(self) -> Llama4TextModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "sdpa"
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }

        return input


class Llama4ForCausalLMBase(ModelClassBase):
    rtol, atol = 1e-2, 1e-2

    def __init__(self):
        super().__init__(Llama4ForCausalLM, llama_config)
        self.row_layers_to_check = [
            "model.layers[0].self_attn.q_proj",
            "model.embed_tokens",
        ]
        self.col_layers_to_check = ["model.layers[0].self_attn.o_proj"]

    def loss_fn(
        self, x: CausalLMOutputWithPast, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor:
        return x.loss

    # Llama4 does not support FlashAttention yet.
    def model_fn(self) -> Llama4ForCausalLM:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "sdpa"
        return self.model_class(config)

    def data_gen_fn(self, num_batch: int) -> dict:
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 256)),
            "attention_mask": torch.ones(num_batch, 256),
        }
        input["labels"] = input["input_ids"]

        return input
