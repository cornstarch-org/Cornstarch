import torch
import torch.distributed as dist
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
)

from ..utils import ModelClassBase

qwen3_config = Qwen3MoeConfig(
    hidden_size=256,
    intermediate_size=64,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=4,
    num_experts=2,
    num_experts_per_tok=1,
    use_cache=False,
)


class Qwen3MoeModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen3MoeModel, qwen3_config)
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


class Qwen3MoeForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(Qwen3MoeForCausalLM, qwen3_config)
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
