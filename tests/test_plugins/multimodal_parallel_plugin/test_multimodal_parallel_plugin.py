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
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
    MultimodalEncoderTrainingZeroBubblePipelineSchedule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_plugin import (
    MultimodalParallelModule,
    MultimodalParallelPlugin,
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
                    [[[0, 0, 1, 1]], [[2, 2, 3, 3]]],
                    [[[4, 4, 5, 5]], [[6, 6, 7, 7]]],
                    [[[8, 9, 10, 11]], [[12, 13, 14, 15]]],
                    [[[16, 17, 18, 19]], [[20, 21, 22, 23]]],
                    [[[24, 25, 26, 27]], [[28, 29, 30, 31]]],
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


class TestMultimodalPluginScheduleSelectionClass(
    TestPluginInitializationWithFakeBackendBase
):
    @staticmethod
    def _build_plugins() -> tuple[ModalParallelPlugin, ModalParallelPlugin]:
        encoder_plugin = ModalParallelPlugin(
            tp_size=1,
            pipeline_template=PipelineTemplate("enc", [["enc"]]),
        )
        llm_plugin = ModalParallelPlugin(
            tp_size=1,
            pipeline_template=PipelineTemplate("llm", [["llm"]]),
        )
        return encoder_plugin, llm_plugin

    def test_pipeline_schedule_string_normalization(self):
        """Accept case-insensitive scheduler names and normalize to lowercase."""
        encoder_plugin, llm_plugin = self._build_plugins()
        plugin = MultimodalParallelPlugin(
            encoder_plugins={"encoder": encoder_plugin},
            language_model_plugin=llm_plugin,
            num_microbatches=8,
            microbatch_size=1,
            pipeline_schedule="ZBPP",
        )
        assert plugin.pipeline_schedule == "zbpp"

    def test_pipeline_schedule_invalid_raises(self):
        """Reject unknown scheduler names with a clear validation error."""
        encoder_plugin, llm_plugin = self._build_plugins()
        with pytest.raises(ValueError, match="pipeline_schedule must be one of"):
            MultimodalParallelPlugin(
                encoder_plugins={"encoder": encoder_plugin},
                language_model_plugin=llm_plugin,
                num_microbatches=8,
                microbatch_size=1,
                pipeline_schedule="invalid",
            )

    @pytest.mark.parametrize(
        "schedule_name,expected_cls",
        [
            ("1f1b", MultimodalEncoderTrainingOneForwardOneBackwardSchedule),
            ("zbpp", MultimodalEncoderTrainingZeroBubblePipelineSchedule),
        ],
    )
    def test_init_distributed_selects_schedule_class(
        self, schedule_name: str, expected_cls: type
    ):
        """Instantiate the correct schedule class during distributed init."""
        encoder_plugin, llm_plugin = self._build_plugins()
        for rank in range(2):
            plugin = MultimodalParallelPlugin(
                encoder_plugins={"encoder": encoder_plugin},
                language_model_plugin=llm_plugin,
                num_microbatches=8,
                microbatch_size=1,
                pipeline_schedule=schedule_name,
            )
            dist.init_process_group(
                backend="fake", store=FakeStore(), rank=rank, world_size=2
            )
            plugin.init_distributed()
            assert isinstance(plugin.schedule, expected_cls)
            dist.destroy_process_group()

