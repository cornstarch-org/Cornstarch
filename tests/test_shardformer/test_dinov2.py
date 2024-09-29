from unittest.mock import patch

import torch
from colossalai.booster import Booster
from colossalai.interface import ModelWrapper, OptimizerWrapper
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    ModelOutput,
)
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Backbone,
    Dinov2Config,
    Dinov2ForImageClassification,
    Dinov2Model,
    Dinov2PreTrainedModel,
)

from cornstarch.shardformer.policies.dinov2 import (
    Dinov2BackbonePolicy,
    Dinov2ForImageClassificationPolicy,
    Dinov2ModelPolicy,
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


class Dinov2PolicyTestClassBase(ColossalaiHybridParallelBase):
    model_class: Dinov2PreTrainedModel
    config = Dinov2Config(
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
        org_model: Dinov2Model,
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
        dino_model = unwrap_model(org_model, "Dinov2Model", "dinov2")
        shard_dino_model = unwrap_model(sharded_model, "Dinov2Model", "dinov2")

        row_layer_for_check = ["encoder.layer[0].attention.attention.query"]
        col_layer_for_check = ["encoder.layer[0].attention.output.dense"]

        atol, rtol = (5e-5, 1e-4) if precision == "fp32" else (5e-3, 5e-3)

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            row_layer_grads = get_grad_tensors_for_check(
                dino_model,
                shard_dino_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                dino_model,
                shard_dino_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(row_layer_grads)
            grads_to_check.update(col_layer_grads)

        # optimizer executes step
        org_optim.step()
        sharded_optim.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            if org_model.__class__.__name__ == "Dinov2Model":
                check_output_hidden_state(
                    org_output, sharded_output, stage_manager, atol=atol, rtol=rtol
                )
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                dino_model,
                shard_dino_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            check_weight(
                dino_model,
                shard_dino_model,
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
class TestDinov2ModelPolicy(Dinov2PolicyTestClassBase):
    def data_gen_fn(self) -> dict:
        pixel_values = torch.randn(1, 3, 224, 224)
        return dict(pixel_values=pixel_values)

    @staticmethod
    def loss_fn(x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    model_class = Dinov2Model

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=Dinov2ModelPolicy(),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)


@instantiate_parametrized_tests
class TestDinov2ForImageClassificationPolicy(Dinov2PolicyTestClassBase):
    def data_gen_fn(self) -> dict:
        data = TestDinov2ModelPolicy.data_gen_fn()
        data["labels"] = torch.tensor([0])
        return data

    @staticmethod
    def loss_fn(x: ImageClassifierOutput) -> torch.Tensor:
        return x.logits.mean()

    model_class = Dinov2ForImageClassification

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=Dinov2ForImageClassificationPolicy(),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)


@instantiate_parametrized_tests
class TestDinov2BackbonePolicy(Dinov2PolicyTestClassBase):
    def data_gen_fn(self) -> dict:
        return TestDinov2ModelPolicy.data_gen_fn()

    @staticmethod
    def loss_fn(x: BackboneOutput) -> torch.Tensor:
        return x.feature_maps[-1].mean()

    model_class = Dinov2Backbone

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=Dinov2BackbonePolicy(),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
