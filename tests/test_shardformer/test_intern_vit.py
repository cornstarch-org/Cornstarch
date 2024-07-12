from unittest.mock import patch

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig
from cornstarch.models.intern_vit.modeling_intern_vit import InternVisionModel
from cornstarch.shardformer.policies.intern_vit import InternVisionModelPolicy

from ._utils import (
    PolicyTestBase,
    build_model_from_hybrid_plugin,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    get_grad_tensors_for_check,
    run_forward_backward_with_hybrid_plugin,
    unwrap_model,
)


@instantiate_parametrized_tests
class TestInternVisionModelPolicyClass(PolicyTestBase):
    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=InternVisionModelPolicy(),
        ):
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_hybrid_plugin(
                model_fn=lambda: InternVisionModel(
                    InternVisionConfig(
                        num_hidden_layers=4,
                        use_flash_attn=False,
                        num_attention_heads=24,  # must be divisible to tp size
                        hidden_size=3072,  # each head has 128 hidden size
                        qk_normalization=False,
                    )
                ),
                loss_fn=lambda x: x["last_hidden_state"].mean(),
                test_config={
                    "tp_size": tp_size,
                    "pp_size": pp_size,
                    "precision": precision,
                    "zero_stage": 0,
                    "num_microbatches": 4,
                    "initial_scale": 1,
                    "enable_flash_attention": fa,
                },
            )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            run_forward_backward_with_hybrid_plugin(
                org_model,
                sharded_model,
                sharded_optimizer,
                lambda: dict(pixel_values=torch.rand(1, 3, 224, 224)),  # data_gen_fn
                lambda x: x,  # output_transform_fn
                criterion,
                booster,
            )
        )

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

        atol, rtol = (5e-5, 1e-4) if precision == "fp32" else (5e-3, 5e-3)

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
        org_optimizer.step()
        sharded_optimizer.step()

        # check last hidden state & loss
        if stage_manager is None or stage_manager.is_last_stage():
            check_output_hidden_state(
                org_output, sharded_output, stage_manager, atol=atol, rtol=rtol
            )
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
