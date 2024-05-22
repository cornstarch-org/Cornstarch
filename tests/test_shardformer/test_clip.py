from unittest.mock import patch

import torch
from conftest import PolicyTestBase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.models.clip.modeling_clip import (
    CLIPVisionConfig,
    CLIPVisionModel,
)

from cornstarch.shardformer.policies.clip import CLIPVisionModelPolicy

from ._utils import (
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
class TestCLIPVisionModelPolicyClass(PolicyTestBase):
    @parametrize("tp_size, pp_size", [(4, 1), (2, 1), (1, 1), (2, 2), (1, 2), (1, 4)])
    @parametrize("fa", [True, False])
    @parametrize("precision", ["fp32", "fp16"])
    def test_hybrid_parallel(
        self, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=CLIPVisionModelPolicy(),
        ):
            (
                org_model,
                org_optimizer,
                sharded_model,
                sharded_optimizer,
                criterion,
                booster,
            ) = build_model_from_hybrid_plugin(
                model_fn=lambda: CLIPVisionModel(CLIPVisionConfig(num_hidden_layers=4)),
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
        clip_model = unwrap_model(org_model, "CLIPVisionModel", "model")
        shard_clip_model = unwrap_model(sharded_model, "CLIPVisionModel", "model")

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
            "vision_model.pre_layrnorm",
        ]

        atol, rtol = (5e-5, 1e-4) if precision == "fp32" else (5e-3, 5e-3)

        # Save gradient tensors for comparison between the original model and the sharded model before optimizer step.
        grads_to_check = {}
        if (
            stage_manager is None or stage_manager.is_first_stage()
        ) and booster.plugin.zero_stage == 0:
            row_layer_grads = get_grad_tensors_for_check(
                clip_model,
                shard_clip_model,
                row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                clip_model,
                shard_clip_model,
                col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            norm_layer_grads = get_grad_tensors_for_check(
                clip_model,
                shard_clip_model,
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
                clip_model,
                shard_clip_model,
                row_layer_for_check,
                tp_group,
                dim=0,
                atol=atol,
                rtol=rtol,
            )
            check_weight(
                clip_model,
                shard_clip_model,
                col_layer_for_check,
                tp_group,
                dim=1,
                atol=atol,
                rtol=rtol,
            )

        check_all_grad_tensors(grads_to_check)
