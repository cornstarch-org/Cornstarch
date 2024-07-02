import copy

import pytest
import torch
import torch.distributed as dist
from colossalai.device import device_mesh
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel

# from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM

from cornstarch.models.multimodal_language_model import (
    ModalModule,
    MultimodalModel,
    MultimodalProjector,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_plugin import (
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)

vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
vision_config.num_hidden_layers = 2
vision_config._attn_implementation = "eager"
language_model_config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
language_model_config.num_hidden_layers = 3
language_model_config._attn_implementation = "eager"

expected_vision_module_layers = [
    "module.vision_model.embeddings",
    "module.vision_model.pre_layrnorm",
    "module.vision_model.encoder.layers.0",
    "module.vision_model.encoder.layers.1",
    "module.vision_model.post_layernorm",
    "projector.projection",
]
expected_language_module_layers = [
    "model.embed_tokens",
    "model.layers.0",
    "model.layers.1",
    "model.layers.2",
    "model.norm",
    "lm_head",
]


def test_initialize_plugin(mocker: MockerFixture):
    mocker.patch.object(
        device_mesh.DeviceMesh,
        "_DIST_BACKEND",
        {"cuda": "nccl", "cpu": "gloo", "npu": "hccl", None: "fake"},
    )

    vision_encoder = CLIPVisionModel(vision_config)
    vision_module = ModalModule(vision_encoder)
    language_module = MistralForCausalLM(language_model_config)

    model = MultimodalModel(
        encoders={"vision": vision_module},
        language_model=language_module,
    ).to(dtype=torch.float16)

    # This check should be done AFTER creating `MultimodalModel`, as it adds a projector inside
    assert PipelineTemplate.get_modules(vision_module) == expected_vision_module_layers
    assert (
        PipelineTemplate.get_modules(language_module) == expected_language_module_layers
    )

    vision_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(vision_module),
            [expected_vision_module_layers[:3], expected_vision_module_layers[3:]],
        ),
    )
    language_plugin = ModalParallelPlugin(
        tp_size=4,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(language_module),
            [
                expected_language_module_layers[:2],
                expected_language_module_layers[2:3],
                expected_language_module_layers[3:],
            ],
        ),
    )

    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_plugin},
        language_model_plugin=language_plugin,
        num_microbatches=4,
        microbatch_size=1,
    )

    world_size = (2 * 2 + 4 * 3) * 2  # dp = 2
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        module = plugin.configure(model)[0]

        assert (
            plugin.stage_manager.pg_mesh.mesh
            == (
                [
                    [[0, 0, 1, 1], [2, 2, 3, 3]],
                    [[4, 4, 5, 5], [6, 6, 7, 7]],
                    [[8, 9, 10, 11], [12, 13, 14, 15]],
                    [[16, 17, 18, 19], [20, 21, 22, 23]],
                    [[24, 25, 26, 27], [28, 29, 30, 31]],
                ]
            )
        ).all()

        assert isinstance(module, MultimodalParallelModule)
        assert module.module.vision_encoder is not None
        assert module.module.language_model is not None
        assert isinstance(module.module.vision_encoder.module, CLIPVisionModel)
        assert isinstance(module.module.vision_encoder.projector, MultimodalProjector)
        assert isinstance(module.module.language_model, MistralForCausalLM)

        dist.destroy_process_group()


def check_layers_cover_all_params(layer_names: list[str], param_names: list[str]):
    used_prefixes = set()
    for param_name in param_names:
        covered = False
        for layer_name in layer_names:
            if param_name.startswith(layer_name):
                used_prefixes.add(layer_name)
                covered = True
                break
        if not covered:
            return False

    return sorted(used_prefixes) == sorted(set(layer_names))


expected_vision_module_params_per_stage = {
    0: [
        "module.vision_model.embeddings",
        "module.vision_model.pre_layrnorm",
        "module.vision_model.encoder.layers.0",
    ],
    1: [
        "module.vision_model.encoder.layers.1",
        "module.vision_model.post_layernorm",
        "projector.projection",
    ],
}
expected_language_module_params_per_stage = {
    2: [
        "model.embed_tokens",
        "model.layers.0",
    ],
    3: [
        "model.layers.1",
    ],
    4: [
        "model.layers.2",
        "model.norm",
        "lm_head",
    ],
}


def test_params_parallelized(mocker: MockerFixture):
    mocker.patch.object(
        device_mesh.DeviceMesh,
        "_DIST_BACKEND",
        {"cuda": "nccl", "cpu": "gloo", "npu": "hccl", None: "fake"},
    )

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
            PipelineTemplate.get_model_name(vision_module),
            [expected_vision_module_layers[:3], expected_vision_module_layers[3:]],
        ),
    )
    language_plugin = ModalParallelPlugin(
        tp_size=4,
        pipeline_template=PipelineTemplate(
            PipelineTemplate.get_model_name(language_module),
            [
                expected_language_module_layers[:2],
                expected_language_module_layers[2:3],
                expected_language_module_layers[3:],
            ],
        ),
    )

    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_plugin},
        language_model_plugin=language_plugin,
        num_microbatches=4,
        microbatch_size=1,
    )

    world_size = (2 * 2 + 4 * 3) * 2  # dp = 2
    stage_indices = {
        (0, 1, 2, 3): 0,
        (4, 5, 6, 7): 1,
        (8, 9, 10, 11, 12, 13, 14, 15): 2,
        (16, 17, 18, 19, 20, 21, 22, 23): 3,
        (24, 25, 26, 27, 28, 29, 30, 31): 4,
    }
    for rank in range(world_size):
        per_rank_plugin = copy.deepcopy(plugin)
        per_rank_model = copy.deepcopy(model)
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        module = per_rank_plugin.configure(per_rank_model)[0]

        stage_index = next(
            stage_index for ranks, stage_index in stage_indices.items() if rank in ranks
        )

        if stage_index in expected_vision_module_params_per_stage:
            assert check_layers_cover_all_params(
                expected_vision_module_params_per_stage[stage_index],
                list(
                    name for name, _ in module.module.vision_encoder.named_parameters()
                ),
            )
        else:
            assert len(list(module.module.vision_encoder.named_parameters())) == 0

        if stage_index in expected_language_module_params_per_stage:
            assert check_layers_cover_all_params(
                expected_language_module_params_per_stage[stage_index],
                list(
                    name for name, _ in module.module.language_model.named_parameters()
                ),
            )
        else:
            assert len(list(module.module.language_model.named_parameters())) == 0

        dist.destroy_process_group()
