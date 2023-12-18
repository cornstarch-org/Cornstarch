import itertools
import sys
from unittest.mock import patch

import numpy as np
import torch.distributed as dist
from data_builder import GLUEDataBuilder
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from transformers import (
    AutoConfig,
    GPT2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from colossalai.interface import ModelWrapper
from colossalai.nn.optimizer import CPUAdam
from colossalai.shardformer.modeling.gpt2 import GPT2PipelineForwards
from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)

# templates are currently based on GPT-2.
# TODO: test more models
config = AutoConfig.from_pretrained("gpt2")
config.num_hidden_layers = 4

homogeneous_templates = {
    PipelineTemplate(
        3,
        2,
        [
            [
                "transformer.wte",
                "transformer.wpe",
                "transformer.drop",
                "transformer.h.0",
            ],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "score"],
        ],
    ): 3
}
heterogeneous_templates = {
    PipelineTemplate(
        3,
        2,
        [
            [
                "transformer.wte",
                "transformer.wpe",
                "transformer.drop",
                "transformer.h.0",
            ],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "score"],
        ],
    ): 1,
    PipelineTemplate(
        2,
        2,
        [
            ["transformer.wte", "transformer.wpe", "transformer.drop"],
            [f"transformer.h.{i}" for i in range(0, 4)]
            + [f"transformer.ln_f", "score"],
        ],
    ): 3,
}


class TestHeterogeneousParallelPluginClass(MultiProcessTestCase):
    pp_axis = 1

    @property
    def world_size(self):
        return 18

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        print(f"dist init r={self.rank}, world={self.world_size}")
        backend = "gloo"

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

            raise

        dist.barrier()

        self.run_test(test_name, parent_pipe)

        dist.barrier()
        dist.destroy_process_group()

    @parametrize(
        "pipeline_templates, expected_mesh, expected_num_stages",
        [
            [
                homogeneous_templates,
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
                {tuple(list(range(18))): 3},
            ],
            [
                heterogeneous_templates,
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
                {tuple(list(range(6))): 3, tuple(list(range(6, 18))): 2},
            ],
        ],
        name_fn=lambda pipeline_templates, expected_mesh, expected_num_stages: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_plugin_initialize(
        self,
        pipeline_templates: dict[PipelineTemplate, int],
        expected_mesh: list,
        expected_num_stages: dict[tuple[int], int],
    ):
        plugin = HeterogeneousParallelPlugin(
            tp_size=2,
            microbatch_size=1,
            num_microbatches=[3] * sum(pipeline_templates.values()),
        )
        plugin.set_pipeline_templates(pipeline_templates)

        assert (
            plugin.shard_config.enable_tensor_parallelism
            and plugin.shard_config.tensor_parallel_size == 2
        )
        assert np.array_equal(plugin.stage_manager.pg_mesh.mesh, expected_mesh)
        assert (
            expected_num_stages[
                next(
                    ranks for ranks in expected_num_stages.keys() if self.rank in ranks
                )
            ]
            == plugin.stage_manager.num_stages
        )

    @parametrize(
        "pipeline_templates, expected_pipeline_index",
        [
            [
                homogeneous_templates,
                {
                    tuple(list(range(0, 6))): 0,
                    tuple(list(range(6, 12))): 1,
                    tuple(list(range(12, 18))): 2,
                },
            ],
            [
                heterogeneous_templates,
                {
                    tuple(list(range(0, 6))): 0,
                    tuple(list(range(6, 10))): 1,
                    tuple(list(range(10, 14))): 2,
                    tuple(list(range(14, 18))): 3,
                },
            ],
        ],
        name_fn=lambda pipeline_templates, _: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_plugin_configure(
        self,
        pipeline_templates: dict[PipelineTemplate, int],
        expected_pipeline_index: dict[tuple[int], int],
    ):
        plugin = HeterogeneousParallelPlugin(
            tp_size=2,
            microbatch_size=1,
            num_microbatches=[0] * sum(pipeline_templates.values()),
        )
        plugin.set_pipeline_templates(pipeline_templates)

        global config
        model = GPT2ForSequenceClassification(config)

        optimizer = CPUAdam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, _, _, lr_scheduler = plugin.configure(
            model,
            optimizer,
            criterion=lambda outputs, inputs: outputs.loss,
            lr_scheduler=lr_scheduler,
        )

        pipeline_index = expected_pipeline_index[
            next(
                ranks for ranks in expected_pipeline_index.keys() if self.rank in ranks
            )
        ]
        assert plugin._pipeline_index == pipeline_index

        assert isinstance(model, ModelWrapper)

        # Check whether the model is split as pipeline template intended
        param_names = list(name for name, _ in model.module.named_parameters())
        pipeline_template = plugin._pipeline_index_to_pipeline[pipeline_index]
        expected_module_names = pipeline_template.modules_per_stage[
            plugin.stage_manager.stage
        ]
        # Get parameters in expected modules after filtering out parameter-less modules
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
                    if list(model.module.get_submodule(module_name).named_parameters())
                ]
            )
        )
        assert param_names == expected_param_names

        # check forward is patched
        assert (
            model.module.forward.func
            is GPT2PipelineForwards.gpt2_for_sequence_classification_forward
        )

    @parametrize(
        "pipeline_templates, unused",
        [[homogeneous_templates, 0], [heterogeneous_templates, 0]],
        name_fn=lambda pipeline_templates, _: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_execute_pipeline(
        self, pipeline_templates: dict[PipelineTemplate, int], unused
    ):
        plugin = HeterogeneousParallelPlugin(
            tp_size=2,
            microbatch_size=1,
            num_microbatches=[4] * sum(pipeline_templates.values()),
        )
        plugin.set_pipeline_templates(pipeline_templates)

        databuilder = GLUEDataBuilder(
            "gpt2",
            plugin,
            train_batch_size=4,
        )
        dataloader = databuilder.train_dataloader()

        global config
        config.num_labels = databuilder.num_labels
        model = GPT2ForSequenceClassification(config)

        optimizer = CPUAdam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, criterion, _, lr_scheduler = plugin.configure(
            model,
            optimizer,
            criterion=lambda outputs, inputs: outputs.loss,
            lr_scheduler=lr_scheduler,
        )

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

            dist.barrier()

        assert "loss" in outputs
        assert mock.called


instantiate_parametrized_tests(TestHeterogeneousParallelPluginClass)
