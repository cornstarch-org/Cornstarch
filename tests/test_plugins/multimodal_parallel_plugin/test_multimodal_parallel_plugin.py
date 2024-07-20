import copy
import os
import random
import sys
from typing import Type
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from colossalai.accelerator import CudaAccelerator
from colossalai.device import device_mesh
from pytest_mock import MockerFixture
from torch.optim import Adam
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed.fake_pg import FakeStore
from transformers import PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
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
    config.hidden_size = 256
    config.intermediate_size = 256
    config.num_attention_heads = 8
    if hasattr(config, "word_embed_proj_dim"):  # for opt
        config.word_embed_proj_dim = 256
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


def generate_multimodal_model(
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
    ).to(dtype=torch.bfloat16)

    return vision_module, language_module, model


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

    @pytest.mark.parametrize("vision_config", vision_configs, ids=["clip"])
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
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
        ids=["tp=(2, 4)"],
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
        vision_module, language_module, model = generate_multimodal_model(
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
                vision_model_name,
                language_model_name,
                vision_tp_size,
                language_tp_size,
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

    @pytest.mark.parametrize("vision_config", vision_configs, ids=["clip"])
    @pytest.mark.parametrize(
        "language_model_config", language_configs, ids=["mistral", "llama", "opt"]
    )
    @pytest.mark.parametrize(
        "world_size, vision_tp_size, language_tp_size, stage_indices",
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
        vision_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        world_size: int,
        vision_tp_size: int,
        language_tp_size: int,
        stage_indices: dict[tuple[int], int],
    ):
        vision_model_name = vision_config[0]
        language_model_name = language_model_config[0]
        *_, model = generate_multimodal_model(
            vision_config[1],
            vision_config[2],
            language_model_config[1],
            language_model_config[2],
        )

        for rank in range(world_size):
            plugin = self.generate_multimodal_plugin(
                vision_model_name,
                language_model_name,
                vision_tp_size=vision_tp_size,
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


class TestParallelPluginExecution(MultiProcessTestCase):
    language_module_layers_per_stage = {
        "mistralai/Mistral-7B-v0.3": [
            ["model.embed_tokens", "model.layers.0"],
            ["model.layers.1", "model.layers.2", "model.norm", "lm_head"],
        ],
        "meta-llama/Meta-Llama-3-8B": [
            ["model.embed_tokens", "model.layers.0"],
            ["model.layers.1", "model.layers.2", "model.norm", "lm_head"],
        ],
        "facebook/opt-125m": [
            [
                "model.decoder.embed_tokens",
                "model.decoder.embed_positions",
                "model.decoder.layers.0",
            ],
            [
                "model.decoder.layers.1",
                "model.decoder.layers.2",
                "model.decoder.final_layer_norm",
                "lm_head",
            ],
        ],
    }

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
                self.language_module_layers_per_stage[language_model_name],
            ),
        )

        return MultimodalParallelPlugin(
            encoder_plugins={"vision": vision_plugin},
            language_model_plugin=language_plugin,
            num_microbatches=4,
            microbatch_size=1,
        )

    @property
    def world_size(self):
        return 4

    def setUp(self) -> None:
        super().setUp()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}):
            self._spawn_processes()

    def tearDown(self) -> None:
        return super().tearDown()

    def get_data(self) -> dict[str, torch.Tensor]:
        input = {
            "pixel_values": torch.from_numpy(
                np.random.rand(12, 3, 224, 224).astype(np.float32)
            ),
            "input_ids": torch.from_numpy(np.random.randint(0, 2048, (12, 64))),
        }
        input["labels"] = input["input_ids"]

        return BatchFeature(input, tensor_type="pt")

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")
        backend = "nccl"

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=backend,
                world_size=self.world_size,
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        device_ids = None
        if torch.cuda.is_available() and torch.cuda.device_count():
            device_id = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            device_ids = [device_id]

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # Execute barrier prior to running test to ensure that every process
        # has finished initialization and that the following test
        # immediately exiting due to a skip doesn't cause flakiness.
        dist.barrier(device_ids=device_ids)

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=True
        ):
            self.run_test(test_name, parent_pipe)

        dist.barrier(device_ids=device_ids)

        dist.destroy_process_group()

    @parametrize(
        "vision_config",
        vision_configs,
        name_fn=lambda x: ["clip"][vision_configs.index(x)],
    )
    @parametrize(
        "language_model_config",
        language_configs,
        name_fn=lambda x: ["mistral", "llama", "opt"][language_configs.index(x)],
    )
    def test_model_run(
        self,
        vision_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
        language_model_config: tuple[str, PretrainedConfig, Type[PreTrainedModel]],
    ):
        vision_model_name = vision_config[0]
        language_model_name = language_model_config[0]
        *_, model = generate_multimodal_model(
            vision_config[1],
            vision_config[2],
            language_model_config[1],
            language_model_config[2],
        )
        model.gradient_checkpointing_enable()
        model.to(device=torch.device("cuda"))

        model_optimizer = Adam(model.parameters())

        inputs = self.get_data()
        for k, v in inputs.items():
            inputs[k] = v.to("cuda")

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

        model_optimizer.step()
        model_optimizer.zero_grad()

        plugin = self.generate_multimodal_plugin(
            vision_model_name,
            language_model_name,
            vision_tp_size=1,
            language_tp_size=1,
        )

        module = copy.deepcopy(model)
        parallel_module_optimizer = Adam(module.parameters())

        module, parallel_module_optimizer, *_ = plugin.configure(
            module,
            parallel_module_optimizer,
        )

        def criterion(x, *args, **kwargs):
            return x.loss

        plugin.execute_pipeline(
            data_iter=iter(inputs),
            model=module,
            criterion=criterion,
            optimizer=parallel_module_optimizer,
            return_loss=True,
        )


instantiate_parametrized_tests(TestParallelPluginExecution)
