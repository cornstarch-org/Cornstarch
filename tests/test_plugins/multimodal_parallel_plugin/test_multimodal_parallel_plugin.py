import copy
from typing import Type

import pytest
import torch
import torch.distributed as dist
from colossalai.device import device_mesh
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral import MistralConfig, MistralForCausalLM
from transformers.models.opt import OPTConfig, OPTForCausalLM

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

clip_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32")
vision_configs = [("openai/clip-vit-base-patch32", clip_config, CLIPVisionModel)]
for _, config, _ in vision_configs:
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"

expected_vision_module_layers = {
    "openai/clip-vit-base-patch32": [
        "module.vision_model.embeddings",
        "module.vision_model.pre_layrnorm",
        "module.vision_model.encoder.layers.0",
        "module.vision_model.encoder.layers.1",
        "module.vision_model.post_layernorm",
        "projector.projection",
    ]
}

mistral_config = MistralConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
llama_config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")
opt_config = OPTConfig.from_pretrained("facebook/opt-125m")

language_configs = [
    ("mistralai/Mistral-7B-v0.3", mistral_config, MistralForCausalLM),
    ("meta-llama/Meta-Llama-3-8B", llama_config, LlamaForCausalLM),
    ("facebook/opt-125m", opt_config, OPTForCausalLM),
]
for _, config, _ in language_configs:
    config.num_hidden_layers = 3
    config._attn_implementation = "eager"

expected_language_module_layers = {
    "mistralai/Mistral-7B-v0.3": [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.norm",
        "lm_head",
    ],
    "meta-llama/Meta-Llama-3-8B": [
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
        "model.layers.2",
        "model.norm",
        "lm_head",
    ],
    "facebook/opt-125m": [
        "model.decoder.embed_tokens",
        "model.decoder.embed_positions",
        "model.decoder.layers.0",
        "model.decoder.layers.1",
        "model.decoder.layers.2",
        "model.decoder.final_layer_norm",
        "lm_head",
    ],
}

expected_vision_module_layers_per_stage = {
    "openai/clip-vit-base-patch32": [
        [
            "module.vision_model.embeddings",
            "module.vision_model.pre_layrnorm",
            "module.vision_model.encoder.layers.0",
        ],
        [
            "module.vision_model.encoder.layers.1",
            "module.vision_model.post_layernorm",
            "projector.projection",
        ],
    ]
}

expected_language_module_layers_per_stage = {
    "mistralai/Mistral-7B-v0.3": [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ],
    "meta-llama/Meta-Llama-3-8B": [
        ["model.embed_tokens", "model.layers.0"],
        ["model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ],
    "facebook/opt-125m": [
        [
            "model.decoder.embed_tokens",
            "model.decoder.embed_positions",
            "model.decoder.layers.0",
        ],
        ["model.decoder.layers.1"],
        ["model.decoder.layers.2", "model.decoder.final_layer_norm", "lm_head"],
    ],
}


class TestPluginInitializationWithFakeBackend:
    @pytest.fixture(autouse=True)
    def fake_backend(self, mocker: MockerFixture):
        mocker.patch.object(
            device_mesh.DeviceMesh,
            "_DIST_BACKEND",
            {"cuda": "nccl", "cpu": "gloo", "npu": "hccl", None: "fake"},
        )
        yield

        if dist.is_initialized():
            dist.destroy_process_group()

    def generate_multimodal_model(
        self,
        vision_config: PretrainedConfig,
        vision_model_cls: Type[PreTrainedModel],
        language_model_config: PretrainedConfig,
        language_model_cls: Type[PreTrainedModel],
    ) -> tuple[ModalModule, PreTrainedModel, MultimodalModel]:
        vision_encoder = vision_model_cls(vision_config)
        vision_module = ModalModule(vision_encoder)
        language_module = language_model_cls(language_model_config)

        model = MultimodalModel(
            encoders={"vision": vision_module},
            language_model=language_module,
        ).to(dtype=torch.float16)

        return vision_module, language_module, model

    def generate_multimodal_plugin(
        self,
        vision_model_name: str,
        language_model_name: str,
        vision_tp_size: int,
        language_tp_size: int,
    ) -> MultimodalParallelPlugin:
        vision_plugin = ModalParallelPlugin(
            tp_size=vision_tp_size,
            pipeline_template=PipelineTemplate(
                vision_model_name,
                expected_vision_module_layers_per_stage[vision_model_name],
            ),
        )
        language_plugin = ModalParallelPlugin(
            tp_size=language_tp_size,
            pipeline_template=PipelineTemplate(
                language_model_name,
                expected_language_module_layers_per_stage[language_model_name],
            ),
        )

        return MultimodalParallelPlugin(
            encoder_plugins={"vision": vision_plugin},
            language_model_plugin=language_plugin,
            num_microbatches=12,
            microbatch_size=1,
        )

    @pytest.mark.parametrize("vision_config", vision_configs)
    @pytest.mark.parametrize("language_model_config", language_configs)
    @pytest.mark.parametrize(
        "world_size, vision_tp_size, language_tp_size, expected_mesh",
        [
            (
                32,
                2,
                4,
                [
                    [[0, 0, 1, 1], [2, 2, 3, 3]],
                    [[4, 4, 5, 5], [6, 6, 7, 7]],
                    [[8, 9, 10, 11], [12, 13, 14, 15]],
                    [[16, 17, 18, 19], [20, 21, 22, 23]],
                    [[24, 25, 26, 27], [28, 29, 30, 31]],
                ],
            )
        ],
    )
    def test_initialize_plugin(
        self,
        vision_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        vision_tp_size: int,
        language_tp_size: int,
        expected_mesh: list[list[list[int]]],
    ):
        vision_model_name = vision_config[0]
        language_model_name = language_model_config[0]
        vision_module, language_module, model = self.generate_multimodal_model(
            vision_config[1],
            vision_config[2],
            language_model_config[1],
            language_model_config[2],
        )

        # This check should be done AFTER creating `MultimodalModel`, as it adds a projector inside
        assert (
            PipelineTemplate.get_modules(vision_module)
            == expected_vision_module_layers[vision_model_name]
        )
        assert (
            PipelineTemplate.get_modules(language_module)
            == expected_language_module_layers[language_model_name]
        )

        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                vision_model_name, language_model_name, vision_tp_size, language_tp_size
            )
            per_rank_model = copy.deepcopy(model)
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            module = plugin.configure(per_rank_model)[0]

            assert (plugin.stage_manager.pg_mesh.mesh == expected_mesh).all()

            assert isinstance(module, MultimodalParallelModule)
            assert module.module.vision_encoder is not None
            assert module.module.language_model is not None
            assert isinstance(module.module.vision_encoder.module, vision_config[2])
            assert isinstance(
                module.module.vision_encoder.projector, MultimodalProjector
            )
            assert isinstance(module.module.language_model, language_model_config[2])

            dist.destroy_process_group()

    def check_layers_cover_all_params(
        self, layer_names: list[str], param_names: list[str]
    ):
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

    @pytest.mark.parametrize("vision_config", vision_configs)
    @pytest.mark.parametrize("language_model_config", language_configs)
    def test_model_parallelization(
        self,
        vision_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
    ):
        vision_model_name = vision_config[0]
        language_model_name = language_model_config[0]
        *_, model = self.generate_multimodal_model(
            vision_config[1],
            vision_config[2],
            language_model_config[1],
            language_model_config[2],
        )

        world_size = 32
        stage_indices = {
            (0, 1, 2, 3): 0,
            (4, 5, 6, 7): 1,
            (8, 9, 10, 11, 12, 13, 14, 15): 2,
            (16, 17, 18, 19, 20, 21, 22, 23): 3,
            (24, 25, 26, 27, 28, 29, 30, 31): 4,
        }
        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                vision_model_name,
                language_model_name,
                vision_tp_size=2,
                language_tp_size=4,
            )

            per_rank_model = copy.deepcopy(model)
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            module, *_ = plugin.configure(per_rank_model)

            stage_index = next(
                stage_index
                for ranks, stage_index in stage_indices.items()
                if rank in ranks
            )

            if stage_index < 2:
                # must only have vision
                assert len(list(module.module.language_model.named_parameters())) == 0
                assert self.check_layers_cover_all_params(
                    expected_vision_module_layers_per_stage[vision_model_name][
                        stage_index
                    ],
                    list(
                        name
                        for name, _ in module.module.vision_encoder.named_parameters()
                    ),
                )
            else:
                # must only have language model
                assert len(list(module.module.vision_encoder.named_parameters())) == 0
                assert self.check_layers_cover_all_params(
                    expected_language_module_layers_per_stage[language_model_name][
                        stage_index - 2
                    ],
                    list(
                        name
                        for name, _ in module.module.language_model.named_parameters()
                    ),
                )

            dist.destroy_process_group()
