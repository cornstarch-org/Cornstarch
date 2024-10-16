import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioEncoderConfig,
)

from ._utils import (
    ColossalaiHybridParallelBase,
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
)


@instantiate_parametrized_tests
class TestQwen2AudioEncoderPolicy(ColossalaiHybridParallelBase):
    model_class = Qwen2AudioEncoder
    config = Qwen2AudioEncoderConfig(
        encoder_attention_heads=4,
        encoder_layers=4,
        is_encoder_decoder=False,
        d_model=384,  # FlashAttention cannot support default value of 1280
    )

    @staticmethod
    def loss_fn(x: BaseModelOutput) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
        )

    def data_gen_fn(self) -> dict:
        batch_size = self.num_microbatches * self.microbatch_size
        return dict(
            input_features=torch.rand(batch_size, self.config.num_mel_bins, 3000)
        )

    def check_fn(
        self,
        booster: Booster,
        org_model: PreTrainedModel,
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
        model: Qwen2AudioEncoder = org_model
        shard_model: Qwen2AudioEncoder = sharded_model.unwrap()

        col_layer_for_check = ["layers[0].self_attn.q_proj"]
        row_layer_for_check = ["layers[0].self_attn.out_proj"]

        atol, rtol = (2e-4, 2e-4) if precision == "fp32" else (5e-3, 5e-3)

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if stage_manager is None or stage_manager.is_first_stage():
            row_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                model,
                shard_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            grads_to_check.update(col_layer_grads)
            grads_to_check.update(row_layer_grads)

        # optimizer executes step
        org_optim.step()
        sharded_optim.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        atol, rtol = (1e-3, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                model,
                shard_model,
                row_layer_for_check,
                tp_group,
                dim=1,
                atol=atol,
                rtol=rtol,
            )
            check_weight(
                model,
                shard_model,
                col_layer_for_check,
                tp_group,
                dim=0,
                atol=atol,
                rtol=rtol,
            )

        check_all_grad_tensors(grads_to_check)

    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp32"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
