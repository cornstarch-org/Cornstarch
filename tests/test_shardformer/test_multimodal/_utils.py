from abc import ABC, abstractmethod
import copy
from typing import Any, Callable, Type

import torch

from .._utils import PolicyTestBase

from colossalai.booster import Booster
from torch.optim import Adam, Optimizer
from colossalai.interface import OptimizerWrapper
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from torch import nn
from cornstarch.plugin.multimodal_parallel_plugin import (
    MultimodalParallelPlugin,
    ModalParallelPlugin,
    MultimodalParallelModule,
)
from cornstarch.models.multimodal_language_model import MultimodalModel, ModalModule
from cornstarch.pipeline_template import PipelineTemplate

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

config_class_dict: dict[str, PretrainedConfig] = {
    "llama": llama_config,
    "gemma": gemma_config,
    "gemma2": gemma2_config,
    "mistral": mistral_config,
    "phi3": phi3_config,
    "qwen2": qwen2_config,
    "clip_vision_model": clip_config,
    "siglip_vision_model": siglip_config,
    "dinov2": dinov2_config,
}

model_class_dict: dict[str, Type[PreTrainedModel]] = {
    "llama": LlamaForCausalLM,
    "gemma": GemmaForCausalLM,
    "gemma2": Gemma2ForCausalLM,
    "mistral": MistralForCausalLM,
    "phi3": Phi3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "clip_vision_model": CLIPVisionModel,
    "siglip_vision_model": SiglipVisionModel,
    "dinov2": Dinov2Model,
}

llama_gemma_mistral_qwen_modules = [
    [
        "module.embed_tokens",
        "module.layers.0",
        "module.layers.1",
        "module.layers.2",
        "module.layers.3",
        "module.norm",
        "lm_head",
    ]
]
phi3_modules = [
    [
        "module.embed_tokens",
        "module.embed_dropout",
        "module.layers.0",
        "module.layers.1",
        "module.layers.2",
        "module.layers.3",
        "module.norm",
        "lm_head",
    ]
]
clip_vision_model_modules = [
    [
        "vision_model.embeddings",
        "vision_model.pre_layrnorm",
        "vision_model.encoder.layers.0",
        "vision_model.encoder.layers.1",
        "vision_model.encoder.layers.2",
        "vision_model.post_layernorm",
    ]
]
siglip_vision_model_modules = [
    [
        "vision_model.embeddings",
        "vision_model.encoder.layers.0",
        "vision_model.encoder.layers.1",
        "vision_model.encoder.layers.2",
        "vision_model.post_layernorm",
        "vision_model.head",
    ]
]
dinov2_modules = [
    [
        "dinov2.embeddings",
        "dinov2.encoder.layers.0",
        "dinov2.encoder.layers.1",
        "dinov2.encoder.layers.2",
        "dinov2.layernorm",
    ]
]

pipeline_template_dict: dict[tuple[str, int], PipelineTemplate] = {
    ("llama", 1): PipelineTemplate("llama", llama_gemma_mistral_qwen_modules),
    ("llama", 2): PipelineTemplate(
        "llama",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("gemma", 1): PipelineTemplate("gemma", llama_gemma_mistral_qwen_modules),
    ("gemma", 2): PipelineTemplate(
        "gemma",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("gemma2", 1): PipelineTemplate("gemma2", llama_gemma_mistral_qwen_modules),
    ("gemma2", 2): PipelineTemplate(
        "gemma2",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("mistral", 1): PipelineTemplate("mistral", llama_gemma_mistral_qwen_modules),
    ("mistral", 2): PipelineTemplate(
        "mistral",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("qwen2", 1): PipelineTemplate("qwen2", llama_gemma_mistral_qwen_modules),
    ("qwen2", 2): PipelineTemplate(
        "qwen2",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("qwen2", 1): PipelineTemplate("qwen2", llama_gemma_mistral_qwen_modules),
    ("qwen2", 2): PipelineTemplate(
        "qwen2",
        [llama_gemma_mistral_qwen_modules[:2], llama_gemma_mistral_qwen_modules[2:]],
    ),
    ("phi3", 1): PipelineTemplate("phi3", phi3_modules),
    ("phi3", 2): PipelineTemplate("phi3", [phi3_modules[:3], phi3_modules[2:]]),
    ("clip_vision_model", 1): PipelineTemplate("clip", clip_vision_model_modules),
    ("clip_vision_model", 2): PipelineTemplate(
        "clip",
        [clip_vision_model_modules[:3], clip_vision_model_modules[3:]],
    ),
    ("siglip_vision_model", 1): PipelineTemplate("siglip", siglip_vision_model_modules),
    ("siglip_vision_model", 2): PipelineTemplate(
        "siglip",
        [siglip_vision_model_modules[:2], siglip_vision_model_modules[2:]],
    ),
    ("dinov2", 1): PipelineTemplate("dinov2", dinov2_modules),
    ("dinov2", 2): PipelineTemplate("dinov2", [dinov2_modules[:2], dinov2_modules[2:]]),
}


def build_model_from_muiltimodal_plugin(
    model_fn: Callable[[], MultimodalModel],
    loss_fn: Callable,
    test_config: dict[str, Any],
) -> tuple[
    MultimodalModel,
    Adam,
    MultimodalParallelModule,
    OptimizerWrapper,
    Callable,
    Booster,
]:
    precision = test_config.pop("precision")
    precision = torch.float32 if precision == "fp32" else torch.float16
    org_model = model_fn().to(dtype=precision, device=torch.device("cuda"))
    sharded_model = copy.deepcopy(org_model)

    org_model = org_model.to("cuda")
    org_optimizer = Adam(org_model.parameters(), lr=1e-3)
    sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)
    criterion = loss_fn

    vision_pp_size = test_config.pop("vision_pp_size")
    language_pp_size = test_config.pop("language_pp_size")
    vision_plugin = ModalParallelPlugin(
        tp_size=test_config.pop("vision_tp_size"),
        pipeline_template=pipeline_template_dict[
            (org_model.vision_encoder.config[0].model_type, vision_pp_size)
        ],
    )
    language_plugin = ModalParallelPlugin(
        tp_size=test_config.pop("language_tp_size"),
        pipeline_template=pipeline_template_dict[
            (org_model.language_model.config.model_type, language_pp_size)
        ],
    )
    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_plugin},
        language_model_plugin=language_plugin,
        precision=None,
        **test_config,
    )
    booster = Booster(plugin=plugin)

    sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
        sharded_model, sharded_optimizer, criterion
    )

    return (
        org_model,
        org_optimizer,
        sharded_model,
        sharded_optimizer,
        criterion,
        booster,
    )


def run_forward_backward_with_multimodal_plugin(
    org_model: nn.Module,
    sharded_model: nn.Module,
    sharded_optimizer: OptimizerWrapper,
    data_gen_fn: Callable[[], dict[str, torch.Tensor]],
    criterion: Callable[[torch.Tensor], torch.Tensor],
    output_transform_fn: Callable,
    booster: Booster,
) -> tuple[
    torch.Tensor, BaseModelOutputWithPast, torch.Tensor, BaseModelOutputWithPast
]:
    org_model.to("cuda")
    sharded_model.to("cuda")

    def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
        outputs = output_transform_fn(outputs)
        loss = criterion(outputs)
        return loss

    data = data_gen_fn()

    shard_test_data = {}
    for k, v in data.items():
        shard_test_data[k] = v.clone().to("cuda")
    unshard_test_data = {}
    for k, v in data.items():
        unshard_test_data[k] = v.clone().to("cuda")

    data_iter = iter([shard_test_data])
    sharded_output = booster.execute_pipeline(
        data_iter,
        sharded_model,
        _criterion,
        sharded_optimizer,
        return_loss=True,
        return_outputs=True,
    )
    sharded_loss = sharded_output["loss"]

    org_output = org_model(**unshard_test_data)
    org_loss = criterion(org_output)
    org_loss.backward()

    return org_loss, org_output, sharded_loss, sharded_output


class VisionLanguagePolicyTestClassBase(PolicyTestBase, ABC):
    vision_model_class: Type[PreTrainedModel]
    language_model_class: Type[PreTrainedModel]
    vision_config: PretrainedConfig
    language_config: PretrainedConfig

    @staticmethod
    @abstractmethod
    def data_gen_fn() -> dict: ...

    @staticmethod
    @abstractmethod
    def loss_fn(x: ModelOutput) -> torch.Tensor: ...

    @abstractmethod
    def check_fn(
        self,
        booster: Booster,
        org_model: MultimodalModel,
        sharded_model: MultimodalParallelModule,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ): ...

    def model_fn(self) -> MultimodalModel:
        vision_config = copy.deepcopy(self.vision_config)
        vision_config.pad_token_id = vision_config.eos_token_id
        language_config = copy.deepcopy(self.language_config)
        language_config.pad_token_id = language_config.eos_token_id

        vision_module = self.vision_model_class(vision_config)
        language_module = self.language_model_class(language_config)

        return MultimodalModel(
            encoders={"vision": ModalModule(vision_module)},
            language_model=language_module,
        )

    def run_multimodal_parallel(
        self,
        vision_tp_size: int,
        vision_pp_size: int,
        language_tp_size: int,
        language_pp_size: int,
        fa: bool,
        precision: str,
    ):
        test_config = {
            "vision_tp_size": vision_tp_size,
            "language_tp_size": language_tp_size,
            "vision_pp_size": vision_pp_size,
            "language_pp_size": language_pp_size,
            "precision": precision,
            "num_microbatches": 4,
            "microbatch_size": 1,
            "initial_scale": 1,
            "enable_flash_attention": fa,
        }

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = build_model_from_muiltimodal_plugin(
            model_fn=self.model_fn,
            loss_fn=self.loss_fn,
            test_config=test_config,
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        org_loss, org_output, sharded_loss, sharded_output = (
            run_forward_backward_with_multimodal_plugin(
                org_model,
                sharded_model,
                sharded_optimizer,
                self.data_gen_fn,
                criterion,
                lambda x: x,  # output_transform_fn,
                booster,
            )
        )

        self.check_fn(
            booster,
            org_model,
            sharded_model,
            org_optimizer,
            sharded_optimizer,
            org_output,
            sharded_output,
            org_loss,
            sharded_loss,
        )
