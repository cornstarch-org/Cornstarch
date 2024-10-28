import copy
from typing import Type

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
from transformers import PretrainedConfig, PreTrainedModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin
from cornstarch.plugin.multimodal_sequential_plugin.multimodal_sequential_plugin import (
    EncoderCoalescedMultimodalParallelModule,
    EncoderCoalescedMultimodalParallelPlugin,
)

from ..common import (
    TestPluginInitializationWithFakeBackendBase,
    audio_configs,
    expected_audio_module_layers,
    expected_audio_module_layers_per_stage,
    expected_language_module_layers,
    expected_language_module_layers_per_stage,
    expected_vision_module_layers,
    expected_vision_module_layers_per_stage,
    language_configs,
    vision_configs,
)


class TestMultiEncoderCoalescedModelInitializationClass(
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
    ) -> EncoderCoalescedMultimodalParallelPlugin:
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

        return EncoderCoalescedMultimodalParallelPlugin(
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
                20,  #  2+3=5 stages, tp=2, dp=2
                (2, 2, 2),
                [
                    [[[0, 1]], [[2, 3]]],
                    [[[4, 5]], [[6, 7]]],
                    [[[8, 9]], [[10, 11]]],
                    [[[12, 13]], [[14, 15]]],
                    [[[16, 17]], [[18, 19]]],
                ],
            ),
            (
                16,  # encoders: 2 stages, tp=2, llm: 3 stages, tp=4
                (2, 2, 4),
                [
                    [[[0, 0, 1, 1]]],
                    [[[2, 2, 3, 3]]],
                    [[[4, 5, 6, 7]]],
                    [[[8, 9, 10, 11]]],
                    [[[12, 13, 14, 15]]],
                ],
            ),
        ],
        ids=["tp=(2, 2, 2)", "tp=(2, 2, 4)"],
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

            assert isinstance(module, EncoderCoalescedMultimodalParallelModule)
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
