from abc import ABC, abstractmethod
from unittest.mock import patch

import torch
from _utils import (
    build_model_from_hybrid_plugin,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)
from colossalai.shardformer.policies.base_policy import Policy
from conftest import PolicyTestBase
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


class Dinov2PolicyTestClassBase(PolicyTestBase, ABC):
    # Implementation for data_gen_fn and loss_fn
    # Copied from colossalai/tests/kit/model_zoo/transformers/vit.py
    @staticmethod
    @abstractmethod
    def data_gen_fn() -> dict: ...

    @staticmethod
    @abstractmethod
    def loss_fn(x: ModelOutput) -> torch.Tensor: ...

    model_class: Dinov2PreTrainedModel
    config = Dinov2Config(
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=4,
    )

    def model_fn(self) -> Dinov2PreTrainedModel:
        return self.model_class(self.config)

    def run_hybrid_parallel(
        self, tp_size: int, pp_size: int, base_model_class_name: str
    ):
        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = build_model_from_hybrid_plugin(
            model_fn=self.model_fn,
            loss_fn=self.loss_fn,
            test_config={
                "tp_size": tp_size,
                "pp_size": pp_size,
                "precision": "fp32",
                "zero_stage": 0,
                "num_microbatches": 4,
            },
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            run_forward_backward_with_hybrid_plugin(
                org_model,
                sharded_model,
                sharded_optimizer,
                self.data_gen_fn,
                lambda x: x,  # output_transform_fn,
                criterion,
                booster,
            )
        )

        stage_manager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group

        # unwrap model
        dino_model = unwrap_model(org_model, base_model_class_name, "dinov2")
        shard_dino_model = unwrap_model(sharded_model, base_model_class_name, "dinov2")

        row_layer_for_check = ["encoder.layer[0].attention.attention.query"]
        col_layer_for_check = ["encoder.layer[0].attention.output.dense"]

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
                atol=2e-5,
                rtol=1e-3,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                dino_model,
                shard_dino_model,
                col_layer_for_check,
                tp_group,
                atol=2e-5,
                rtol=1e-3,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(row_layer_grads)
            grads_to_check.update(col_layer_grads)

        # optimizer executes step
        org_optimizer.step()
        sharded_optimizer.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            if org_model.__class__.__name__ == "Dinov2Model":
                check_output_hidden_state(
                    org_output, sharded_output, stage_manager, atol=2e-3, rtol=1e-3
                )
            check_loss(org_loss, sharded_loss, atol=2e-3, rtol=1e-3)

        # check weights
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                dino_model,
                shard_dino_model,
                row_layer_for_check,
                tp_group,
                atol=5e-3,
                rtol=1e-3,
                dim=0,
                verbose=False,
            )
            check_weight(
                dino_model,
                shard_dino_model,
                col_layer_for_check,
                tp_group,
                atol=5e-3,
                rtol=1e-3,
                dim=1,
                verbose=False,
            )

        # check grads
        check_all_grad_tensors(grads_to_check)


class TestDinov2ModelPolicy(Dinov2PolicyTestClassBase):
    @staticmethod
    def data_gen_fn() -> dict:
        pixel_values = torch.randn(1, 3, 224, 224)
        return dict(pixel_values=pixel_values)

    @staticmethod
    def loss_fn(x: BaseModelOutputWithPooling) -> torch.Tensor:
        return x.pooler_output.mean()

    model_class = Dinov2Model

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    def test_hybrid_parallel(self, tp_size: int, pp_size: int):
        with (
            patch(
                "colossalai.shardformer.shard.sharder.get_autopolicy",
                return_value=Dinov2ModelPolicy(),
            ),
            patch.object(
                Dinov2ModelPolicy,
                "distribute_layers",
                new=lambda _, *args: Policy.distribute_layers(*args),
            ),
            patch.object(
                Dinov2ModelPolicy,
                "get_stage_index",
                new=lambda _, *args: Policy.get_stage_index(*args),
            ),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, "Dinov2Model")


class TestDinov2ForImageClassificationPolicy(Dinov2PolicyTestClassBase):
    @staticmethod
    def data_gen_fn() -> dict:
        data = TestDinov2ModelPolicy.data_gen_fn()
        data["labels"] = torch.tensor([0])
        return data

    @staticmethod
    def loss_fn(x: ImageClassifierOutput) -> torch.Tensor:
        return x.logits.mean()

    model_class = Dinov2ForImageClassification

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    def test_hybrid_parallel(self, tp_size: int, pp_size: int):
        with (
            patch(
                "colossalai.shardformer.shard.sharder.get_autopolicy",
                return_value=Dinov2ForImageClassificationPolicy(),
            ),
            patch.object(
                Dinov2ForImageClassificationPolicy,
                "distribute_layers",
                new=lambda _, *args: Policy.distribute_layers(*args),
            ),
            patch.object(
                Dinov2ForImageClassificationPolicy,
                "get_stage_index",
                new=lambda _, *args: Policy.get_stage_index(*args),
            ),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, "Dinov2Model")


class TestDinov2BackbonePolicy(Dinov2PolicyTestClassBase):
    @staticmethod
    def data_gen_fn() -> dict:
        return TestDinov2ModelPolicy.data_gen_fn()

    @staticmethod
    def loss_fn(x: BackboneOutput) -> torch.Tensor:
        return x.feature_maps[-1].mean()

    model_class = Dinov2Backbone

    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    def test_hybrid_parallel(self, tp_size: int, pp_size: int):
        with (
            patch(
                "colossalai.shardformer.shard.sharder.get_autopolicy",
                return_value=Dinov2BackbonePolicy(),
            ),
            patch.object(
                Dinov2BackbonePolicy,
                "distribute_layers",
                new=lambda _, *args: Policy.distribute_layers(*args),
            ),
            patch.object(
                Dinov2BackbonePolicy,
                "get_stage_index",
                new=lambda _, *args: Policy.get_stage_index(*args),
            ),
        ):
            self.run_hybrid_parallel(tp_size, pp_size, "Dinov2Backbone")


instantiate_parametrized_tests(TestDinov2ModelPolicy)
instantiate_parametrized_tests(TestDinov2ForImageClassificationPolicy)
instantiate_parametrized_tests(TestDinov2BackbonePolicy)
