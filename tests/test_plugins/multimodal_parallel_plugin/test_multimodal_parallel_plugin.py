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
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_plugin import (
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)

whisper_config: WhisperConfig = WhisperConfig.from_pretrained("openai/whisper-small")
whisper_config.encoder_layers = 2
whisper_config._attn_implementation = "eager"
audio_configs = [("openai/whisper-small", whisper_config, WhisperEncoder)]

expected_audio_module_layers = {
    "openai/whisper-small": [
        "module.conv1",
        "module.conv2",
        "module.embed_positions",
        "module.layers.0",
        "module.layers.1",
        "module.layer_norm",
        "projector.projection",
    ]
}

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
    config.hidden_size = 256
    config.intermediate_size = 256
    config.num_attention_heads = 8
    if hasattr(config, "word_embed_proj_dim"):  # for opt
        config.word_embed_proj_dim = 256
    config._attn_implementation = "eager"
    config.tie_word_embeddings = False # opt uses it

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

expected_audio_module_layers_per_stage = {
    "openai/whisper-small": [
        [
            "module.conv1",
            "module.conv2",
            "module.embed_positions",
            "module.layers.0",
        ],
        [
            "module.layers.1",
            "module.layer_norm",
            "projector.projection",
        ],
    ]
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


class TestPluginInitializationWithFakeBackendBase:
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


class TestSingleEncoderModelInitializationClass(
    TestPluginInitializationWithFakeBackendBase
):
    def generate_multimodal_model(
        self,
        encoder_config: PretrainedConfig,
        encoder_model_cls: Type[PreTrainedModel],
        language_model_config: PretrainedConfig,
        language_model_cls: Type[PreTrainedModel],
    ) -> tuple[ModalEncoderModule, PreTrainedModel, MultimodalModel]:
        encoder_module = encoder_model_cls(encoder_config)
        encoder_module = ModalEncoderModule(encoder_module)
        language_module = language_model_cls(language_model_config)

        model = MultimodalModel(
            encoders={"encoder": encoder_module},
            language_model=language_module,
        ).to(dtype=torch.float16)

        return encoder_module, language_module, model

    def generate_multimodal_plugin(
        self,
        encoder_model_name: str,
        language_model_name: str,
        encoder_tp_size: int,
        language_tp_size: int,
    ) -> MultimodalParallelPlugin:
        encoder_layers_per_stage = (
            expected_audio_module_layers_per_stage[encoder_model_name]
            if encoder_model_name in expected_audio_module_layers_per_stage
            else expected_vision_module_layers_per_stage[encoder_model_name]
        )
        encoder_plugin = ModalParallelPlugin(
            tp_size=encoder_tp_size,
            pipeline_template=PipelineTemplate(
                encoder_model_name, encoder_layers_per_stage
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
            encoder_plugins={"encoder": encoder_plugin},
            language_model_plugin=language_plugin,
            num_microbatches=12,
            microbatch_size=1,
        )

    @pytest.mark.parametrize(
        "encoder_config", audio_configs + vision_configs, ids=["whisper", "clip"]
    )
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
    @pytest.mark.parametrize(
        "world_size, encoder_tp_size, language_tp_size, expected_mesh",
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
        ids=["tp=(2, 4)"],
    )
    def test_initialize_plugin(
        self,
        encoder_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        encoder_tp_size: int,
        language_tp_size: int,
        expected_mesh: list[list[list[int]]],
    ):
        encoder_model_name = encoder_config[0]
        language_model_name = language_model_config[0]
        encoder_module, language_module, model = self.generate_multimodal_model(
            encoder_config[1],
            encoder_config[2],
            language_model_config[1],
            language_model_config[2],
        )

        # This check should be done AFTER creating `MultimodalModel`, as it adds a projector inside
        if encoder_model_name in expected_audio_module_layers_per_stage:
            assert (
                PipelineTemplate.get_modules(encoder_module)
                == expected_audio_module_layers[encoder_model_name]
            )
        else:
            assert (
                PipelineTemplate.get_modules(encoder_module)
                == expected_vision_module_layers[encoder_model_name]
            )
        assert (
            PipelineTemplate.get_modules(language_module)
            == expected_language_module_layers[language_model_name]
        )

        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                encoder_model_name,
                language_model_name,
                encoder_tp_size,
                language_tp_size,
            )
            per_rank_model = copy.deepcopy(model)
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            module = plugin.configure(per_rank_model)[0]

            assert (plugin.stage_manager.pg_mesh.mesh == expected_mesh).all()

            assert isinstance(module, MultimodalParallelModule)
            assert module.module.encoder_encoder is not None
            assert module.module.language_model is not None
            assert isinstance(module.module.encoder_encoder.module, encoder_config[2])
            assert isinstance(
                module.module.encoder_encoder.projector, MultimodalProjector
            )
            assert isinstance(module.module.language_model, language_model_config[2])

            dist.destroy_process_group()

    @pytest.mark.parametrize(
        "encoder_config", audio_configs + vision_configs, ids=["whisper", "clip"]
    )
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
    @pytest.mark.parametrize(
        "world_size, encoder_tp_size, language_tp_size, stage_indices",
        [
            (
                32,
                2,
                4,
                {
                    (0, 1, 2, 3): 0,
                    (4, 5, 6, 7): 1,
                    (8, 9, 10, 11, 12, 13, 14, 15): 2,
                    (16, 17, 18, 19, 20, 21, 22, 23): 3,
                    (24, 25, 26, 27, 28, 29, 30, 31): 4,
                },
            ),
            (
                20,
                4,
                4,
                {
                    (0, 1, 2, 3): 0,
                    (4, 5, 6, 7): 1,
                    (8, 9, 10, 11): 2,
                    (12, 13, 14, 15): 3,
                    (16, 17, 18, 19): 4,
                },
            ),
            (
                10,
                1,
                1,
                {
                    (0, 1): 0,
                    (2, 3): 1,
                    (4, 5): 2,
                    (6, 7): 3,
                    (8, 9): 4,
                },
            ),
        ],
        ids=["tp=(2, 4)", "tp=(4, 4)", "tp=(1, 1)"],
    )
    def test_model_parallelization(
        self,
        encoder_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        encoder_tp_size: int,
        language_tp_size: int,
        stage_indices: dict[tuple[int], int],
    ):
        encoder_model_name = encoder_config[0]
        language_model_name = language_model_config[0]
        *_, model = self.generate_multimodal_model(
            encoder_config[1],
            encoder_config[2],
            language_model_config[1],
            language_model_config[2],
        )

        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                encoder_model_name,
                language_model_name,
                encoder_tp_size=encoder_tp_size,
                language_tp_size=language_tp_size,
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
                # must only have encoder
                assert len(list(module.module.language_model.named_parameters())) == 0
                parameters_in_encoder = list(
                    name for name, _ in module.module.encoder_encoder.named_parameters()
                )
                if encoder_model_name in expected_audio_module_layers_per_stage:
                    assert self.check_layers_cover_all_params(
                        expected_audio_module_layers_per_stage[encoder_model_name][
                            stage_index
                        ],
                        parameters_in_encoder,
                    )
                else:
                    assert self.check_layers_cover_all_params(
                        expected_vision_module_layers_per_stage[encoder_model_name][
                            stage_index
                        ],
                        parameters_in_encoder,
                    )
            else:
                # must only have language model
                assert len(list(module.module.encoder_encoder.named_parameters())) == 0
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


class TestMultiEncodersModelInitializationClass(
    TestPluginInitializationWithFakeBackendBase
):
    def generate_multimodal_model(
        self,
        encoder_configs: list[PretrainedConfig],
        encoder_model_clss: list[Type[PreTrainedModel]],
        language_model_config: PretrainedConfig,
        language_model_cls: Type[PreTrainedModel],
    ) -> tuple[dict[str, ModalEncoderModule], PreTrainedModel, MultimodalModel]:
        assert len(encoder_configs) == len(encoder_model_clss)
        encoder_modules = {}
        for index, (encoder_config, encoder_model_cls) in enumerate(
            zip(encoder_configs, encoder_model_clss)
        ):
            encoder_module = encoder_model_cls(encoder_config)
            encoder_module = ModalEncoderModule(encoder_module)
            encoder_modules[f"encoder{index}"] = encoder_module

        language_module = language_model_cls(language_model_config)

        model = MultimodalModel(
            encoders=encoder_modules,
            language_model=language_module,
        ).to(dtype=torch.float16)

        return encoder_modules, language_module, model

    def generate_multimodal_plugin(
        self,
        encoder_model_names: list[str],
        language_model_name: str,
        encoder_tp_sizes: list[int],
        language_tp_size: int,
    ) -> MultimodalParallelPlugin:
        assert len(encoder_model_names) == len(encoder_tp_sizes)
        encoder_plugins = {}
        for index, (encoder_model_name, encoder_tp_size) in enumerate(
            zip(encoder_model_names, encoder_tp_sizes)
        ):
            encoder_layers_per_stage = (
                expected_audio_module_layers_per_stage[encoder_model_name]
                if encoder_model_name in expected_audio_module_layers_per_stage
                else expected_vision_module_layers_per_stage[encoder_model_name]
            )
            encoder_plugin = ModalParallelPlugin(
                tp_size=encoder_tp_size,
                pipeline_template=PipelineTemplate(
                    encoder_model_name, encoder_layers_per_stage
                ),
            )
            encoder_plugins[f"encoder{index}"] = encoder_plugin

        language_plugin = ModalParallelPlugin(
            tp_size=language_tp_size,
            pipeline_template=PipelineTemplate(
                language_model_name,
                expected_language_module_layers_per_stage[language_model_name],
            ),
        )

        return MultimodalParallelPlugin(
            encoder_plugins=encoder_plugins,
            language_model_plugin=language_plugin,
            num_microbatches=12,
            microbatch_size=1,
        )

    @pytest.mark.parametrize("vision_model_config", vision_configs, ids=["clip"])
    @pytest.mark.parametrize("audio_model_config", audio_configs, ids=["whisper"])
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
    @pytest.mark.parametrize(
        "world_size, tp_size, expected_mesh",
        [
            (
                28,
                (2, 2, 2),
                [
                    [[0, 1], [2, 3]],
                    [[4, 5], [6, 7]],
                    [[8, 9], [10, 11]],
                    [[12, 13], [14, 15]],
                    [[16, 17], [18, 19]],
                    [[20, 21], [22, 23]],
                    [[24, 25], [26, 27]],
                ],
            ),
            (
                20,
                (2, 2, 4),
                [
                    [[0, 0, 1, 1]],
                    [[2, 2, 3, 3]],
                    [[4, 4, 5, 5]],
                    [[6, 6, 7, 7]],
                    [[8, 9, 10, 11]],
                    [[12, 13, 14, 15]],
                    [[16, 17, 18, 19]],
                ],
            ),
            (
                18,
                (2, 4, 2),
                [
                    [[0, 0, 1, 1]],
                    [[2, 2, 3, 3]],
                    [[4, 5, 6, 7]],
                    [[8, 9, 10, 11]],
                    [[12, 12, 13, 13]],
                    [[14, 14, 15, 15]],
                    [[16, 16, 17, 17]],
                ],
            ),
        ],
        ids=["tp=(2, 2, 2)", "tp=(2, 2, 4)", "tp=(2, 4, 2)"],
    )
    def test_initialize_plugin(
        self,
        vision_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        audio_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        tp_size: tuple[int, int, int],
        expected_mesh: list[list[list[int]]],
    ):
        vision_model_name = vision_model_config[0]
        audio_model_name = audio_model_config[0]
        language_model_name = language_model_config[0]
        encoder_modules, language_module, model = self.generate_multimodal_model(
            [vision_model_config[1], audio_model_config[1]],
            [vision_model_config[2], audio_model_config[2]],
            language_model_config[1],
            language_model_config[2],
        )

        assert (
            PipelineTemplate.get_modules(encoder_modules["encoder0"])
            == expected_vision_module_layers[vision_model_name]
        )
        assert (
            PipelineTemplate.get_modules(encoder_modules["encoder1"])
            == expected_audio_module_layers[audio_model_name]
        )
        assert (
            PipelineTemplate.get_modules(language_module)
            == expected_language_module_layers[language_model_name]
        )

        vision_tp_size, audio_tp_size, language_tp_size = tp_size
        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                [vision_model_name, audio_model_name],
                language_model_name,
                [vision_tp_size, audio_tp_size],
                language_tp_size,
            )
            per_rank_model = copy.deepcopy(model)
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            module = plugin.configure(per_rank_model)[0]

            assert (plugin.stage_manager.pg_mesh.mesh == expected_mesh).all()

            assert isinstance(module, MultimodalParallelModule)
            assert module.module.encoder0_encoder is not None
            assert module.module.encoder1_encoder is not None
            assert module.module.language_model is not None

            assert isinstance(
                module.module.encoder0_encoder.module, vision_model_config[2]
            )
            assert isinstance(
                module.module.encoder1_encoder.module, audio_model_config[2]
            )
            assert isinstance(
                module.module.encoder0_encoder.projector, MultimodalProjector
            )
            assert isinstance(
                module.module.encoder1_encoder.projector, MultimodalProjector
            )
            assert isinstance(module.module.language_model, language_model_config[2])

            dist.destroy_process_group()

    @pytest.mark.parametrize("vision_model_config", vision_configs, ids=["clip"])
    @pytest.mark.parametrize("audio_model_config", audio_configs, ids=["whisper"])
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
    @pytest.mark.parametrize(
        "world_size, tp_size, stage_indices",
        [
            (
                28,
                (2, 2, 2),
                {
                    (0, 1, 2, 3): 0,
                    (4, 5, 6, 7): 1,
                    (8, 9, 10, 11): 2,
                    (12, 13, 14, 15): 3,
                    (16, 17, 18, 19): 4,
                    (20, 21, 22, 23): 5,
                    (24, 25, 26, 27): 6,
                },
            ),
            (
                20,
                (2, 2, 4),
                {
                    (0, 1): 0,
                    (2, 3): 1,
                    (4, 5): 2,
                    (6, 7): 3,
                    (8, 9, 10, 11): 4,
                    (12, 13, 14, 15): 5,
                    (16, 17, 18, 19): 6,
                },
            ),
            (
                18,
                (2, 4, 2),
                {
                    (0, 1): 0,
                    (2, 3): 1,
                    (4, 5, 6, 7): 2,
                    (8, 9, 10, 11): 3,
                    (12, 13): 4,
                    (14, 15): 5,
                    (16, 17): 6,
                },
            ),
        ],
        ids=["tp=(2, 2, 2)", "tp=(2, 2, 4)", "tp=(2, 4, 2)"],
    )
    def test_modal_parallelization(
        self,
        vision_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        audio_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        tp_size: tuple[int, int, int],
        stage_indices: dict[tuple[int], int],
    ):
        vision_model_name = vision_model_config[0]
        audio_model_name = audio_model_config[0]
        language_model_name = language_model_config[0]
        encoder_modules, language_module, model = self.generate_multimodal_model(
            [vision_model_config[1], audio_model_config[1]],
            [vision_model_config[2], audio_model_config[2]],
            language_model_config[1],
            language_model_config[2],
        )

        vision_tp_size, audio_tp_size, language_tp_size = tp_size
        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                [vision_model_name, audio_model_name],
                language_model_name,
                [vision_tp_size, audio_tp_size],
                language_tp_size,
            )
            per_rank_model = copy.deepcopy(model)
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=world_size
            )
            module = plugin.configure(per_rank_model)[0]

            stage_index = next(
                stage_index
                for ranks, stage_index in stage_indices.items()
                if rank in ranks
            )

            if stage_index < 2:
                # must only have vision encoder
                assert len(list(module.module.encoder1_encoder.named_parameters())) == 0
                assert len(list(module.module.language_model.named_parameters())) == 0
                assert self.check_layers_cover_all_params(
                    expected_vision_module_layers_per_stage[vision_model_name][
                        stage_index
                    ],
                    list(
                        name
                        for name, _ in module.module.encoder0_encoder.named_parameters()
                    ),
                )
            elif stage_index < 4:
                # must only have audio encoder
                assert len(list(module.module.encoder0_encoder.named_parameters())) == 0
                assert len(list(module.module.language_model.named_parameters())) == 0
                assert self.check_layers_cover_all_params(
                    expected_audio_module_layers_per_stage[audio_model_name][
                        stage_index - 2
                    ],
                    list(
                        name
                        for name, _ in module.module.encoder1_encoder.named_parameters()
                    ),
                )
            else:
                # must only have language model
                assert len(list(module.module.encoder0_encoder.named_parameters())) == 0
                assert len(list(module.module.encoder1_encoder.named_parameters())) == 0
                assert self.check_layers_cover_all_params(
                    expected_language_module_layers_per_stage[language_model_name][
                        stage_index - 4
                    ],
                    list(
                        name
                        for name, _ in module.module.language_model.named_parameters()
                    ),
                )

            dist.destroy_process_group()
