from unittest.mock import patch

import torch
from _utils import (
    build_model,
    check_grad,
    run_forward,
)

# from colossalai.logging import disable_existing_loggers
from colossalai.testing import (
    assert_hf_output_close,
)
from conftest import PolicyTestBase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.models.clip.modeling_clip import (
    CLIPVisionConfig,
    CLIPVisionModel,
)

from cornstarch.shardformer.policies.clip import CLIPVisionPolicy


class TestCLIPVisionPolicyClass(PolicyTestBase):
    def check_forward_backward(
        self, org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn
    ):
        org_output, org_loss, shard_output, shard_loss = run_forward(
            org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn
        )

        assert_hf_output_close(
            org_output.last_hidden_state, shard_output.last_hidden_state
        )

        # do backward
        org_loss.backward()
        shard_loss.backward()

        assert torch.allclose(
            org_loss, shard_loss, atol=1e-5
        ), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

        # check grad
        row_layer_for_check = [
            "vision_model.encoder.layers[0].self_attn.out_proj",
            "vision_model.encoder.layers[0].mlp.fc2",
        ]
        col_layer_for_check = [
            "vision_model.encoder.layers[0].self_attn.q_proj",
            "vision_model.encoder.layers[0].mlp.fc1",
        ]

        check_grad(
            org_model,
            sharded_model,
            col_layer_for_check,
            atol=1e-6,
            rtol=1e-5,
            dim=0,
            verbose=False,
        )
        check_grad(
            org_model,
            sharded_model,
            row_layer_for_check,
            atol=1e-6,
            rtol=1e-5,
            dim=1,
            verbose=False,
        )

    @parametrize("precision", [torch.float32, torch.bfloat16])
    @parametrize("enable_fused_normalization", [True, False])
    @parametrize("enable_flash_attention", [True, False])
    def test_clip_vision(
        self,
        precision: torch.dtype,
        enable_fused_normalization: bool,
        enable_flash_attention: bool,
    ):
        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=CLIPVisionPolicy(),
        ):
            config = CLIPVisionConfig()
            config.attention_dropout = 0.0

            org_model, sharded_model = build_model(
                model_fn=lambda: CLIPVisionModel(config),
                enable_fused_normalization=enable_fused_normalization,
                enable_flash_attention=enable_flash_attention,
                enable_tensor_parallelism=True,
                enable_jit_fused=False,
                dtype=precision,
            )

        self.check_forward_backward(
            org_model,
            sharded_model,
            data_gen_fn=lambda: dict(
                pixel_values=torch.rand(1, 3, 224, 224).to(precision)
            ),
            output_transform_fn=lambda x: x,
            loss_fn=lambda x: x["last_hidden_state"].mean(),
        )


instantiate_parametrized_tests(TestCLIPVisionPolicyClass)

# def check_forward_backward(
#     org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn
# ):
#     org_output, org_loss, shard_output, shard_loss = run_forward(
#         org_model, sharded_model, data_gen_fn, output_transform_fn, loss_fn
#     )
#     assert_hf_output_close(org_output, shard_output, ignore_keys=["past_key_values"])

#     # do backward
#     org_loss.backward()
#     shard_loss.backward()

#     assert torch.allclose(
#         org_loss, shard_loss, atol=1e-5
#     ), f"shard model loss is not equal to orgin model loss\n{org_loss}\n{shard_loss}"

#     # check grad
#     row_layer_for_check = [
#         "vision_model.encoder.layers[0].self_attn.q_proj",
#         "vision_model.encoder.layers[0].mlp.fc1",
#     ]
#     col_layer_for_check = [
#         "vision_model.encoder.layers[0].self_attn.out_proj",
#         "vision_model.encoder.layers[0].mlp.fc2",
#     ]

#     check_grad(
#         org_model,
#         sharded_model,
#         col_layer_for_check,
#         atol=1e-6,
#         rtol=1e-5,
#         dim=0,
#         verbose=False,
#     )
#     check_grad(
#         org_model,
#         sharded_model,
#         row_layer_for_check,
#         atol=1e-6,
#         rtol=1e-5,
#         dim=1,
#         verbose=False,
#     )


# config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")


# def data_gen():
#     pixel_values = torch.rand(1, 3, 224, 224)
#     return dict(pixel_values=pixel_values)


# @parameterize(
#     "test_config",
#     [
#         {
#             "tp_size": 4,
#             "pp_size": 1,
#             "num_microbatches": 4,
#             "precision": torch.float32,
#         },
#         {
#             "tp_size": 4,
#             "pp_size": 1,
#             "num_microbatches": 4,
#             "precision": torch.bfloat16,
#         },
#     ],
# )
# def run_clip_test(test_config: dict):
#     with patch(
#         "colossalai.shardformer.shard.sharder.get_autopolicy",
#         return_value=CLIPVisionPolicy(),
#     ):
#         org_model, sharded_model = build_model(
#             model_fn=lambda: CLIPVisionModel(config),
#             enable_fused_normalization=False,
#             enable_flash_attention=False,
#             enable_tensor_parallelism=True if test_config["tp_size"] > 1 else False,
#             enable_jit_fused=False,
#             dtype=test_config["precision"],
#         )

#     check_forward_backward(
#         org_model,
#         sharded_model,
#         data_gen_fn=data_gen,
#         output_transform_fn=lambda x: x,
#         loss_fn=lambda x: x["last_hidden_state"].mean(),
#     )

#     torch.cuda.empty_cache()


# def check_clip(rank, world_size, port):
#     disable_existing_loggers()
#     colossalai.launch(
#         config={},
#         rank=rank,
#         world_size=world_size,
#         host="localhost",
#         port=port,
#         backend="nccl",
#     )
#     run_clip_test()


# @pytest.mark.dist
# @rerun_if_address_is_in_use()
# @clear_cache_before_run()
# def test_clip_vision():
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
#     spawn(check_clip, 1)


# if __name__ == "__main__":
#     test_clip_vision()
