from __future__ import annotations

import itertools
import sys
from typing import Callable, cast
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.shardformer.modeling.gpt2 import GPT2PipelineForwards
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.heterogeneous_dataloader import HeterogeneousDataLoader
from cornstarch.plugin.heterogeneous_parallel_module import (
    HeterogeneousParallelModule,
)
from cornstarch.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from cornstarch.shardformer.policies.gpt2 import (
    GPT2ForSequenceClassificationPolicy,
)
from cornstarch.shardformer.shard.placeholder import TensorPlaceholder
from data_builder import GLUEDataBuilder
from torch.optim import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from transformers import (
    GPT2Config,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

config: GPT2Config = GPT2Config.from_pretrained("gpt2")
config.is_decoder = True
config.n_layer = 4
config.num_labels = GLUEDataBuilder.glue_task_num_labels["mrpc"]

modules: list[str] = GPT2ForSequenceClassificationPolicy.get_all_modules(config)
model_name: str = "transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification"

template_1stage = PipelineTemplate(model_name, [modules])
template_2stages = PipelineTemplate(model_name, [modules[:3], modules[3:]])
template_3stages = PipelineTemplate(
    model_name, [modules[:4], modules[4:7], modules[7:]]
)


class HeterogeneousParallelPluginClassBase(MultiProcessTestCase):
    num_hosts: int
    tp_size: int
    backend: str

    @property
    def world_size(self) -> int:
        return self.num_hosts * self.tp_size

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def init_distributed(self):
        if self.backend and self.backend == "nccl":
            torch.cuda.set_device(self.rank)
        print(f"dist init r={self.rank}, world={self.world_size}")

        try:
            dist.init_process_group(
                init_method=f"{FILE_SCHEMA}{self.file_name}",
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

    def prepare(
        self, pipelines: list[PipelineTemplate], tp_size: int
    ) -> HeterogeneousParallelPlugin:
        self.init_distributed()
        plugin = HeterogeneousParallelPlugin(
            tp_size=tp_size,
            microbatch_size=1,
        )
        plugin.set_pipelines(
            pipelines=pipelines,
            num_microbatches={template: 6 for template in pipelines},
        )
        return plugin

    def _test_plugin_initialize(
        self,
        pipelines: list[PipelineTemplate],
        expected_num_stages: list[int],
        expected_mesh: list,
    ):
        plugin = self.prepare(pipelines, self.tp_size)

        assert (
            plugin.shard_config.enable_tensor_parallelism
            and plugin.shard_config.tensor_parallel_size == self.tp_size
        )
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)
        assert expected_num_stages[self.rank] == plugin.stage_manager.num_stages

    def configure(
        self, pipelines: list[PipelineTemplate]
    ) -> tuple[
        HeterogeneousParallelPlugin,
        ModelWrapper,
        OptimizerWrapper,
        Callable,
        HeterogeneousDataLoader,
        LRScheduler,
    ]:
        plugin = self.prepare(pipelines, self.tp_size)

        dataloader = GLUEDataBuilder("gpt2", plugin).train_dataloader()
        model = GPT2ForSequenceClassification(config)

        optimizer = Adam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        return plugin, *plugin.configure(
            model,
            optimizer,
            lambda outputs, inputs: outputs.loss,
            dataloader,
            lr_scheduler,
        )

    def _test_plugin_configure(
        self,
        pipelines: list[PipelineTemplate],
        expected_pipeline_index: list[int],
    ):
        plugin, model, *_ = self.configure(pipelines)

        pipeline_index = expected_pipeline_index[self.rank]
        assert plugin._pipeline_index == pipeline_index
        assert isinstance(model, HeterogeneousParallelModule)

        # Check whether the model is split as pipeline template intended
        param_names = list(
            name
            for name, param in model.module.named_parameters()
            if not isinstance(param, TensorPlaceholder)
        )
        pipeline_template = plugin.pipelines[pipeline_index]
        expected_module_names = pipeline_template.modules_per_stage[
            plugin.stage_manager.stage
        ]
        # Get parameters in expected modules
        expected_param_names = list(
            itertools.chain.from_iterable(
                [
                    [
                        f"{module_name}.{name}"
                        for name, _ in model.module.get_submodule(
                            module_name
                        ).named_parameters()
                    ]
                    for module_name in expected_module_names
                ]
            )
        )
        assert param_names == expected_param_names

        # check forward is patched
        assert (
            model.module.forward.func
            is GPT2PipelineForwards.gpt2_for_sequence_classification_forward
        )


class TestHeterogeneous3DParallelPluginConfigurationClass(
    HeterogeneousParallelPluginClassBase
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 9
        self.tp_size = 2
        self.backend = "gloo"

    @parametrize(
        "pipelines, expected_num_stages, expected_mesh",
        [
            [
                [template_3stages, template_3stages, template_3stages],
                [3] * 18,
                [
                    [
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [2, 3],
                        [2, 3],
                        [2, 3],
                        [4, 5],
                        [4, 5],
                    ],
                    [
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [8, 9],
                        [8, 9],
                        [8, 9],
                        [10, 11],
                        [10, 11],
                    ],
                    [
                        [12, 13],
                        [12, 13],
                        [12, 13],
                        [12, 13],
                        [14, 15],
                        [14, 15],
                        [14, 15],
                        [16, 17],
                        [16, 17],
                    ],
                ],
            ],
            [
                [
                    template_3stages,
                    template_2stages,
                    template_2stages,
                    template_2stages,
                ],
                [3] * 6 + [2] * 12,
                [
                    [
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [2, 3],
                        [2, 3],
                        [2, 3],
                        [4, 5],
                        [4, 5],
                    ],
                    [
                        [6, 7],
                        [6, 7],
                        [6, 7],
                        [8, 9],
                        [8, 9],
                        [8, 9],
                        [8, 9],
                        [8, 9],
                        [8, 9],
                    ],
                    [
                        [10, 11],
                        [10, 11],
                        [10, 11],
                        [12, 13],
                        [12, 13],
                        [12, 13],
                        [12, 13],
                        [12, 13],
                        [12, 13],
                    ],
                    [
                        [14, 15],
                        [14, 15],
                        [14, 15],
                        [16, 17],
                        [16, 17],
                        [16, 17],
                        [16, 17],
                        [16, 17],
                        [16, 17],
                    ],
                ],
            ],
        ],
        name_fn=lambda pipelines, *_: "homogeneous"
        if len(set(pipelines)) == 1
        else "heterogeneous",
    )
    def test_plugin_initialte(
        self,
        pipelines: list[PipelineTemplate],
        expected_num_stages: list[int],
        expected_mesh: list,
    ):
        self._test_plugin_initialize(pipelines, expected_num_stages, expected_mesh)

    @parametrize(
        "pipelines, expected_pipeline_index",
        [
            [
                [template_3stages, template_3stages, template_3stages],
                [0] * 6 + [1] * 6 + [2] * 6,
            ],
            [
                [
                    template_3stages,
                    template_2stages,
                    template_2stages,
                    template_2stages,
                ],
                [0] * 6 + [1] * 4 + [2] * 4 + [3] * 4,
            ],
        ],
        name_fn=lambda pipelines, _: "homogeneous"
        if len(set(pipelines)) == 1
        else "heterogeneous",
    )
    def test_plugin_configure(
        self, pipelines: list[PipelineTemplate], expected_pipeline_index: list[int]
    ):
        self._test_plugin_configure(pipelines, expected_pipeline_index)


class TestHomogeneousExecutionClass(HeterogeneousParallelPluginClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 4
        self.tp_size = 1
        self.backend = "nccl"

    @skip_if_lt_x_gpu(4)
    def test_execute(self):
        pipelines = [template_2stages, template_2stages]

        plugin, model, optimizer, criterion, dataloader, lr_scheduler = self.configure(
            pipelines
        )

        model: HeterogeneousParallelModule = cast(HeterogeneousParallelModule, model)

        with patch.object(
            model, "sync_dp_grads", side_effect=model.sync_dp_grads
        ) as mock:
            outputs = plugin.execute_pipeline(
                iter(dataloader),
                model,
                criterion,
                optimizer,
                return_loss=True,
                return_outputs=True,
            )

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            dist.barrier()

        assert "loss" in outputs
        assert mock.called


class TestHeterogeneousExecutionClass(HeterogeneousParallelPluginClassBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hosts = 3
        self.tp_size = 1
        self.backend = "nccl"

    @skip_if_lt_x_gpu(3)
    def test_execute(self):
        pipelines = [template_1stage, template_2stages]

        plugin, model, optimizer, criterion, dataloader, lr_scheduler = self.configure(
            pipelines
        )

        model: HeterogeneousParallelModule = cast(HeterogeneousParallelModule, model)

        with patch.object(
            model, "sync_dp_grads", side_effect=model.sync_dp_grads
        ) as mock:
            outputs = plugin.execute_pipeline(
                iter(dataloader),
                model,
                criterion,
                optimizer,
                return_loss=True,
                return_outputs=True,
            )

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            dist.barrier()

        assert "loss" in outputs
        assert mock.called


instantiate_parametrized_tests(TestHeterogeneous3DParallelPluginConfigurationClass)
