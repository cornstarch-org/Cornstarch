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

from cornstarch.models.internlm2.modeling_internlm2 import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2Model,
    InternLM2PreTrainedModel,
)

from ._utils import (
    ColossalaiHybridParallelBase,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    unwrap_model,
)


class InternLM2PolicyTestClassBase(ColossalaiHybridParallelBase):
    model_class: InternLM2PreTrainedModel
    config = InternLM2Config(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=16,
        num_hidden_layers=4,
        use_cache=False,
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
        org_model: InternLM2Model,
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
        precision = booster.plugin.precision

        # unwrap model
        model = unwrap_model(org_model, "InternLM2Model", "model")
        shard_model = unwrap_model(sharded_model, "InternLM2Model", "model")

        row_layer_for_check = ["layers[0].attention.wqkv", "tok_embeddings"]
        col_layer_for_check = ["layers[0].attention.wo"]

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            atol, rtol = (1e-6, 1e-4) if precision == "fp32" else (5e-3, 5e-3)
            row_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
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
            atol, rtol = (1e-5, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage():
            atol, rtol = (1e-3, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            # embed_tokens have different dimension, so skip to check row_layer weight
            check_weight(
                model,
                shard_model,
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
class TestInternLM2ModelPolicy(InternLM2PolicyTestClassBase):
    @staticmethod
    def loss_fn(x: BaseModelOutputWithPast) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    model_class = InternLM2Model

    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp32"])
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
class TestInternLM2ForCausalLMPolicy(InternLM2PolicyTestClassBase):
    def data_gen_fn(self) -> dict:
        input = super().data_gen_fn()
        input["labels"] = input["input_ids"]

        return input

    @staticmethod
    def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
        return x.loss

    model_class = InternLM2ForCausalLM

    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp32"])
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
