import torch
import torch.distributed as dist
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM, Gemma2Model

from ..utils import ModelClassBase

gemma2_config = Gemma2Config(
    hidden_size=256,
    intermediate_size=64,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=4,
    # TODO: all pretrained Gemma model uses head_dim=256, which is not supported
    # by bitfield attention mask (Triton FlashAttention base).
    head_dim=64,
    use_cache=False,
    # TODO: Gemma uses tie_word_embeddings True, in which case the tests fail.
    # Implement automatic gradient synchronization between tied weights.
    # Existing explicit synchronization is not enough as there are encoders
    # that need to have gradients propagated "after" the weights are synchronized.
    tie_word_embeddings=False,
)


class Gemma2ModelBase(ModelClassBase):
    def __init__(self):
        super().__init__(Gemma2Model, gemma2_config)
        self.col_layers_to_check = ["layers[0].self_attn.o_proj"]
        self.row_layers_to_check = ["layers[0].self_attn.q_proj", "embed_tokens"]

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


class Gemma2ForCausalLMBase(ModelClassBase):
    def __init__(self):
        super().__init__(Gemma2ForCausalLM, gemma2_config)
        self.col_layers_to_check = ["model.layers[0].self_attn.o_proj"]
        self.row_layers_to_check = [
            "model.layers[0].self_attn.q_proj",
            "model.embed_tokens",
        ]

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
