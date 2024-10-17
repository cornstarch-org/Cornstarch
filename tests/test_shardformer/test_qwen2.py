import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen2PreTrainedModel,
)

from ._utils import (
    ColossalaiHybridParallelBase,
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
    unwrap_model,
)


class Qwen2PolicyTestClassBase(ColossalaiHybridParallelBase):
    model_class: Qwen2PreTrainedModel
    config = Qwen2Config(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=4,
        _attn_implementation="eager",
    )

    def data_gen_fn(self) -> dict:
        num_batch = self.num_microbatches * self.microbatch_size
        input = {
            "input_ids": torch.randint(0, 2048, (num_batch, 64)),
            "attention_mask": torch.ones(num_batch, 64),
        }

        return input

    def check_fn(
        self,
        booster: Booster,
        org_model: Qwen2Model,
        sharded_model: ModelWrapper,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        stage_manager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group

        # unwrap model
        qwen2_model = unwrap_model(org_model, "Qwen2Model", "model")
        shard_qwen2_model = unwrap_model(sharded_model, "Qwen2Model", "model")

        row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
        col_layer_for_check = ["layers[0].self_attn.o_proj"]

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            atol, rtol = 5e-3, 5e-3
            row_layer_grads = get_grad_tensors_for_check(
                qwen2_model,
                shard_qwen2_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                qwen2_model,
                shard_qwen2_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(col_layer_grads)
            grads_to_check.update(row_layer_grads)

        # optimizer executes step
        org_optim.step()
        sharded_optim.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            atol, rtol = 5e-3, 5e-3
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage(ignore_chunk=True):
            atol, rtol = 5e-3, 5e-3
            # embed_tokens have different dimension, so skip to check row_layer weight
            check_weight(
                qwen2_model,
                shard_qwen2_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )

        # check grads
        check_all_grad_tensors(grads_to_check)


@instantiate_parametrized_tests
class TestQwen2ModelPolicy(Qwen2PolicyTestClassBase):
    @staticmethod
    def loss_fn(x: BaseModelOutputWithPast) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    model_class = Qwen2Model

    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    def test_context_parallel(self, tp_size: int, pp_size: int):
        # ring_attn is for causal lm only
        self.run_hybrid_parallel(tp_size, pp_size, "all_to_all", True, "bf16")


@instantiate_parametrized_tests
class TestQwen2ForCausalLMPolicy(Qwen2PolicyTestClassBase):
    @staticmethod
    def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    def data_gen_fn(self) -> dict:
        input = super().data_gen_fn()
        input["labels"] = input["input_ids"]

        return input

    model_class = Qwen2ForCausalLM

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("sp_mode", ["all_to_all", "ring_attn"], name_fn=lambda x: x)
    def test_context_parallel(self, tp_size: int, pp_size: int, sp_mode: str):
        self.run_hybrid_parallel(tp_size, pp_size, sp_mode, True, "bf16")
