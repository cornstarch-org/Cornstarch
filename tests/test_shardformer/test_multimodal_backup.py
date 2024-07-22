import copy
import itertools
from typing import Any, Callable, Type
import numpy as np
import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

import functools

from ._utils import (
    PolicyTestBase,
    check_all_grad_tensors,
    check_loss,
    check_output_hidden_state,
    check_weight,
    unwrap_model,
    get_grad_tensors_for_check,
)

from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
import torch.nn as nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.optim import Adam
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.siglip import SiglipVisionConfig, SiglipVisionModel
from transformers.models.dinov2 import Dinov2Config, Dinov2Model

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalParallelPlugin,
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultiModalPipelineStageManager,
)
from cornstarch.models.multimodal_language_model import MultimodalModel, ModalModule
from cornstarch.pipeline_template import PipelineTemplate

llama_config = LlamaConfig()
gemma_config = GemmaConfig()
gemma2_config = Gemma2Config()
mistral_config = MistralConfig()
phi3_config = Phi3Config()
qwen2_config = Qwen2Config()

for language_config in [
    llama_config,
    gemma_config,
    gemma2_config,
    mistral_config,
    phi3_config,
    qwen2_config,
]:
    language_config.hidden_size = 256
    language_config.intermediate_size = 256
    language_config.num_attention_heads = 8
    language_config.num_hidden_layers = 4
    language_config.use_cache = False
    language_config._attn_implementation = "eager"
    language_config.num_key_value_heads = 8

clip_config = CLIPVisionConfig()
siglip_config = SiglipVisionConfig()
dinov2_config = Dinov2Config()

for vision_config in [clip_config, siglip_config, dinov2_config]:
    vision_config.hidden_size = 256
    vision_config.intermediate_size = 256
    vision_config.num_attention_heads = 8
    vision_config.num_hidden_layers = 3
    vision_config.use_cache = False
    vision_config._attn_implementation = "eager"

config_class_dict = {
    "llama": llama_config,
    "gemma": gemma_config,
    "gemma2": gemma2_config,
    "mistral": mistral_config,
    "phi3": phi3_config,
    "qwen2": qwen2_config,
    "clip": clip_config,
    "siglip": siglip_config,
}

model_class_dict = {
    "llama": LlamaForCausalLM,
    "gemma": GemmaForCausalLM,
    "gemma2": Gemma2ForCausalLM,
    "mistral": MistralForCausalLM,
    "phi3": Phi3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "clip": CLIPVisionModel,
    "siglip": SiglipVisionModel,
}

pipeline_templates_2stages: dict[tuple[PretrainedConfig], PipelineTemplate] = {
    (
        "llama",
        "gemma",
        "gemma2",
        "mistral",
        "qwen2",
    ): PipelineTemplate(
        "model",
        [
            ["module.embed_tokens", "module.layers.0"],
            [
                "module.layers.1",
                "module.layers.2",
                "module.layers.3",
                "module.norm",
                "lm_head",
            ],
        ],
    ),
    ("phi3",): PipelineTemplate(
        "model",
        [
            ["module.embed_tokens", "module.embed_dropout", "module.layers.0"],
            [
                "module.layers.1",
                "module.layers.2",
                "module.layers.3",
                "module.norm",
                "lm_head",
            ],
        ],
    ),
    ("clip",): PipelineTemplate(
        "model",
        [
            [
                "vision_model.embeddings",
                "vision_model.pre_layrnorm",
                "vision_model.encoder.layers.0",
            ],
            [
                "vision_model.encoder.layers.1",
                "vision_model.encoder.layers.2",
                "vision_model.post_layernorm",
            ],
        ],
    ),
    ("siglip",): PipelineTemplate(
        "model",
        [
            [
                "vision_model.embeddings",
                "vision_model.encoder.layers.0",
            ],
            [
                "vision_model.encoder.layers.1",
                "vision_model.encoder.layers.2",
                "vision_model.post_layernorm",
                "vision_model.head",
            ],
        ],
    ),
}


def get_pipeline_template(
    config: PretrainedConfig, model_name: str, num_stages: int
) -> PipelineTemplate:
    assert num_stages in (1, 2)
    pipeline_template_2stages: PipelineTemplate = next(
        template
        for model_names, template in pipeline_templates_2stages.items()
        if model_name in model_names
    )
    if num_stages == 2:
        return pipeline_template_2stages
    else:
        return PipelineTemplate(
            pipeline_template_2stages.model_name,
            [list(itertools.chain(*pipeline_template_2stages.modules_per_stage))],
        )


class TestMultimodalModelPolicy(PolicyTestBase):
    @staticmethod
    def data_gen_fn(num_microbatches: int, microbatch_size: int) -> dict:
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
    def loss_fn(outputs: CausalLMOutputWithPast, inputs) -> torch.Tensor:
        return outputs.loss

    def model_fn(
        self, config: PretrainedConfig, model_cls: Type[PreTrainedModel]
    ) -> PreTrainedModel:
        config = copy.deepcopy(config)
        config.pad_token_id = config.eos_token_id
        return model_cls(config)

    @parametrize(
        "vision_model_name",
        ["clip", "siglip", "dinov2"],
        name_fn=lambda v: v,
    )
    @parametrize(
        "language_model_name",
        [
            "llama",
            "gemma",
            "gemma2",
            "mistral",
            "phi3",
            "qwen2",
        ],
        name_fn=lambda l: l,
    )
    @parametrize(
        "vision_tp_size, vision_pp_size, language_tp_size, language_pp_size",
        [(1, 1, 1, 1), (1, 2, 1, 2), (2, 1, 2, 1)],
        name_fn=lambda vtp, vpp, ltp, lpp: f"v=({vtp},{vpp}),l=({ltp},{lpp})",
    )
    @parametrize("precision", ["fp32", "bf16"])
    def test_multimodal_parallel(
        self,
        vision_model_name: str,
        language_model_name: str,
        vision_tp_size: int,
        language_tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        precision: str,
    ):
        vision_config = config_class_dict[vision_model_name]
        language_config = config_class_dict[language_model_name]

        test_config = {
            "vision_tp_size": vision_tp_size,
            "language_tp_size": language_tp_size,
            "vision_pp_size": vision_pp_size,
            "language_pp_size": language_pp_size,
            "precision": precision,
            "num_microbatches": 4,
            "microbatch_size": 1,
            "initial_scale": 1,
        }

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = self.build_model_from_muiltimodal_plugin(
            vision_config=vision_config,
            language_config=language_config,
            vision_model_name=vision_model_name,
            language_model_name=language_model_name,
            test_config=test_config,
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            self.run_forward_backward_with_multimodal_plugin(
                org_model,
                sharded_model,
                sharded_optimizer,
                functools.partial(
                    self.data_gen_fn,
                    test_config["num_microbatches"],
                    test_config["microbatch_size"],
                ),
                criterion,
                booster,
            )
        )

        stage_manager: MultiModalPipelineStageManager = booster.plugin.stage_manager
        tp_group = booster.plugin.tp_group

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
        atol, rtol = (1e-4, 1e-4) if precision == "fp32" else (5e-3, 5e-3)
        if stage_manager.is_first_stage(check_only_in_modal=False):
            # vision first stage
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
        elif stage_manager.is_first_stage(check_only_in_modal=True):
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
        org_optimizer.step()
        sharded_optimizer.step()

        if stage_manager.is_last_stage(check_only_in_modal=False):
            atol, rtol = (1e-5, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
            check_loss(org_loss, sharded_loss, atol=atol, rtol=rtol)

        # check weights
        atol, rtol = (1e-4, 1e-3) if precision == "fp32" else (5e-3, 5e-3)
        if stage_manager.is_first_stage(check_only_in_modal=False):
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
        elif stage_manager.is_first_stage(check_only_in_modal=True):
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


instantiate_parametrized_tests(TestMultimodalModelPolicy)
