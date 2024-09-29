import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
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


@instantiate_parametrized_tests
class TestSiglipVisionModelPolicyClass(ColossalaiHybridParallelBase):
    model_class = SiglipVisionModel
    config = SiglipVisionConfig(
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
    )

    def data_gen_fn(self) -> dict:
        image_size = self.config.image_size
        num_channels = self.config.num_channels
        num_batch = self.num_microbatches * self.microbatch_size
        return {
            "pixel_values": torch.randn(num_batch, num_channels, image_size, image_size)
        }

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
        siglip_model = unwrap_model(org_model, "SiglipVisionModel", "model")
        shard_siglip_model = unwrap_model(sharded_model, "SiglipVisionModel", "model")

        col_layer_for_check = [
            "vision_model.encoder.layers[0].self_attn.out_proj",
            "vision_model.encoder.layers[0].mlp.fc2",
        ]
        row_layer_for_check = [
            "vision_model.encoder.layers[0].self_attn.q_proj",
            "vision_model.encoder.layers[0].mlp.fc1",
        ]
        norm_layer_for_check = [
            "vision_model.encoder.layers[0].layer_norm1",
            "vision_model.encoder.layers[0].layer_norm2",
        ]

        atol, rtol = (5e-5, 1e-4) if precision == "fp32" else (5e-3, 5e-3)

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            row_layer_grads = get_grad_tensors_for_check(
                siglip_model,
                shard_siglip_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                siglip_model,
                shard_siglip_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            norm_layer_grads = get_grad_tensors_for_check(
                siglip_model,
                shard_siglip_model,
                norm_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(col_layer_grads)
            grads_to_check.update(row_layer_grads)
            grads_to_check.update(norm_layer_grads)

        # optimizer executes step
        org_optim.step()
        sharded_optim.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            check_output_hidden_state(
                org_output, sharded_output, stage_manager, atol=atol, rtol=rtol
            )
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                siglip_model,
                shard_siglip_model,
                row_layer_for_check,
                tp_group,
                dim=0,
                atol=atol,
                rtol=rtol,
            )
            check_weight(
                siglip_model,
                shard_siglip_model,
                col_layer_for_check,
                tp_group,
                dim=1,
                atol=atol,
                rtol=rtol,
            )

        check_all_grad_tensors(grads_to_check)

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
