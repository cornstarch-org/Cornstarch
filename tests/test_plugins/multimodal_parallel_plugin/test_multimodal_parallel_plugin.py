import pytest
import torch
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel

# from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM

from cornstarch.models.multimodal_language_model import ModalModule, MultimodalModel
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_plugin import (
    MultimodalParallelPlugin,
)

vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
vision_config.num_hidden_layers = 2
language_model_config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
language_model_config.num_hidden_layers = 3


def test_initialize_plugin():
    vision_encoder = CLIPVisionModel(vision_config)
    vision_module = ModalModule(vision_encoder)
    language_module = MistralForCausalLM(language_model_config)

    model = MultimodalModel(
        encoders={"vision": vision_module},
        language_model=language_module,
    ).to(dtype=torch.float16)

    vision_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(vision_encoder),
            PipelineTemplate.get_modules(vision_encoder),
        ),
    )
    language_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(language_module),
            PipelineTemplate.get_modules(language_module),
        ),
    )

    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_plugin},
        language_model_plugin=language_plugin,
        num_microbatches=4,
        microbatch_size=1,
    )
