import copy

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
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
            [PipelineTemplate.get_modules(vision_encoder)],
        ),
    )
    language_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(language_module),
            [PipelineTemplate.get_modules(language_module)],
        ),
    )

    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_plugin},
        language_model_plugin=language_plugin,
        num_microbatches=4,
        microbatch_size=1,
    )

    world_size = 4
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        plugin.configure(model)
        dist.destroy_process_group()
