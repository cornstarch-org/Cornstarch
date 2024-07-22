import numpy as np
import torch

from ._utils import VisionLanguagePolicyTestClassBase, config_class_dict
from .._utils import (
    get_grad_tensors_for_check,
    check_weight,
    check_loss,
    check_all_grad_tensors,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from colossalai.booster import Booster
from transformers.modeling_outputs import CausalLMOutputWithPast
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalParallelPlugin,
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultiModalPipelineStageManager,
)
from cornstarch.models.multimodal_language_model import MultimodalModel, ModalModule
from torch.optim import Optimizer
from colossalai.interface import OptimizerWrapper

from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.clip import CLIPVisionModel, CLIPVisionConfig


class ClipLlamaForcausalLMPolicyTestClass(VisionLanguagePolicyTestClassBase):
    vision_model_class = CLIPVisionModel
    language_model_class = LlamaForCausalLM
    vision_config: CLIPVisionConfig = config_class_dict["clip_vision_model"]
    language_config: LlamaConfig = config_class_dict["llama"]

    @staticmethod
    def data_gen_fn() -> dict:
        microbatch_size = 1
        num_microbatches = 4
        num_batch = microbatch_size * num_microbatches
        input = {
            "pixel_values": torch.from_numpy(
                np.random.rand(num_batch, 3, 224, 224).astype(np.float32)
            ),
            "input_ids": torch.from_numpy(np.random.randint(0, 2048, (num_batch, 64))),
            "attention_mask": torch.from_numpy(np.ones((num_batch, 64))),
        }
        input["labels"] = input["input_ids"]

        return input

    @staticmethod
    def loss_fn(outputs: CausalLMOutputWithPast) -> torch.Tensor:
        return outputs.loss

    def check_fn(
        self,
        booster: Booster,
        org_model: MultimodalModel,
        sharded_model: MultimodalParallelModule,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: CausalLMOutputWithPast,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        stage_manager: MultiModalPipelineStageManager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group
        precision = next(org_model.parameters()).dtype

        # unwrap_model
        org_vision_model = org_model.encoders["vision"].module.vision_model
        org_language_model = org_model.language_model.model
        sharded_vision_model = sharded_model.module.encoders[
            "vision"
        ].module.vision_model
        sharded_language_model = sharded_model.module.language_model.model

        vision_row_layer_for_check = [
            "encoder.layers[0].self_attn.q_proj",
            "encoder.layers[0].mlp.fc1",
        ]
        vision_col_layer_for_check = [
            "encoder.layers[0].self_attn.out_proj",
            "encoder.layers[0].mlp.fc2",
        ]
        language_row_layer_for_check = ["layers[0].self_attn.q_proj", "embed_tokens"]
        language_col_layer_for_check = ["layers[0].self_attn.o_proj"]

        grads_to_check = {}
        atol, rtol = (1e-4, 1e-4) if precision == torch.float32 else (5e-3, 5e-3)

        is_vision_first_stage = (
            sharded_model.my_modal_name == "vision_encoder"
            and stage_manager.is_first_stage(check_only_in_modal=True)
        )
        is_language_first_stage = (
            sharded_model.my_modal_name == "language_model"
            and stage_manager.is_first_stage(check_only_in_modal=True)
        )

        if is_vision_first_stage:
            row_layer_grads = get_grad_tensors_for_check(
                org_vision_model,
                sharded_vision_model,
                vision_row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                org_vision_model,
                sharded_vision_model,
                vision_col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
            grads_to_check.update(row_layer_grads)
            grads_to_check.update(col_layer_grads)
        elif is_language_first_stage:
            # language first stage
            row_layer_grads = get_grad_tensors_for_check(
                org_language_model,
                sharded_language_model,
                language_row_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=0,
                verbose=False,
            )
            col_layer_grads = get_grad_tensors_for_check(
                org_language_model,
                sharded_language_model,
                language_col_layer_for_check,
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

        if stage_manager.is_last_stage(check_only_in_modal=False):
            atol, rtol = (1e-5, 1e-3) if precision == torch.float32 else (5e-3, 5e-3)
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        atol, rtol = (1e-4, 1e-3) if precision == torch.float32 else (5e-3, 5e-3)
        if is_vision_first_stage:
            check_weight(
                org_vision_model,
                sharded_vision_model,
                vision_col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )
        elif is_language_first_stage:
            # embed_tokens have different dimension, so skip to check row_layer weight
            check_weight(
                org_language_model,
                sharded_language_model,
                language_col_layer_for_check,
                tp_group,
                atol=atol,
                rtol=rtol,
                dim=1,
                verbose=False,
            )

        # check grads
        check_all_grad_tensors(grads_to_check)

    @parametrize(
        "vision_tp_size, vision_pp_size, language_tp_size, language_pp_size",
        [
            (1, 1, 1, 1),
            (1, 2, 1, 2),
            (2, 1, 2, 1),
        ],
        name_fn=lambda vtp, vpp, ltp, lpp: f"v=({vtp},{vpp}),l=({ltp},{lpp})",
    )
    @parametrize("precision", ["bf16", "fp32"])
    @parametrize("fa", [False])
    def test_clip_llama_causallm(
        self,
        vision_tp_size: int,
        vision_pp_size: int,
        language_tp_size: int,
        language_pp_size: int,
        fa: bool,
        precision: str,
    ):
        self.run_multimodal_parallel(
            vision_tp_size,
            vision_pp_size,
            language_tp_size,
            language_pp_size,
            fa,
            precision,
        )


instantiate_parametrized_tests(ClipLlamaForcausalLMPolicyTestClass)
