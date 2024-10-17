import copy

import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig
from cornstarch.models.intern_vit.modeling_intern_vit import InternVisionModel

from ._utils import (
    ColossalaiHybridParallelBase,
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
    unwrap_model,
)


@instantiate_parametrized_tests
class TestInternVisionModelPolicyClass(ColossalaiHybridParallelBase):
    model_class = InternVisionModel
    config = InternVisionConfig(
        intermediate_size=128,
        num_hidden_layers=4,
        use_flash_attn=False,
        num_attention_heads=16,  # must be divisible to tp size
        hidden_size=2048,  # each head has 128 hidden size
        qk_normalization=False,
    )

    # HF does not provide InternVision flash attention yet.
    # Use SDPA implementation and compare against ColoAttention.
    def model_fn(self, fa: bool) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "eager"
        return self.model_class(config)

    @staticmethod
    def loss_fn(x: BaseModelOutputWithPooling) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            x.last_hidden_state, torch.ones_like(x.last_hidden_state)
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
        org_model: InternVisionModel,
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
        intern_model = unwrap_model(org_model, "InternVisionModel", "model")
        shard_intern_model = unwrap_model(sharded_model, "InternVisionModel", "model")

        row_layer_for_check = [
            "encoder.layers[0].attn.qkv",
            "encoder.layers[0].mlp.fc1",
        ]
        col_layer_for_check = [
            "encoder.layers[0].attn.proj",
            "encoder.layers[0].mlp.fc2",
        ]
        norm_layer_for_check = [
            "encoder.layers[0].norm1",
            "encoder.layers[0].norm2",
        ]

        atol, rtol = 5e-3, 5e-3

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            row_layer_grads = get_grad_tensors_for_check(
                intern_model,
                shard_intern_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                intern_model,
                shard_intern_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            norm_layer_grads = get_grad_tensors_for_check(
                intern_model,
                shard_intern_model,
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
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                intern_model,
                shard_intern_model,
                row_layer_for_check,
                tp_group,
                dim=0,
                atol=atol,
                rtol=rtol,
            )
            check_weight(
                intern_model,
                shard_intern_model,
                col_layer_for_check,
                tp_group,
                dim=1,
                atol=atol,
                rtol=rtol,
            )

        check_all_grad_tensors(grads_to_check)

    @parametrize("tp_size, pp_size", [(4, 1), (1, 1), (2, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
