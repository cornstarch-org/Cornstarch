import sys

import numpy as np
import torch.distributed as dist
from colossalai.interface import ModelWrapper
from colossalai.nn.optimizer import CPUAdam
from torch.testing._internal.common_distributed import TEST_SKIPS, MultiProcessTestCase
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    instantiate_parametrized_tests,
    parametrize,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)


# templates are currently based on GPT-2.
# TODO: test more models
config = AutoConfig.from_pretrained("gpt2")
config.num_hidden_layers = 4

homogeneous_templates = {
    PipelineTemplate(
        config,
        ["0"],
        [2, 2, 2],
        [
            [f"transformer.{emb}" for emb in ["wte", "wpe", "drop"]]
            + [f"transformer.h.0"],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "lm_head"],
        ],
        [
            "transformer.wte",
            "transformer.wpe",
            "transformer.drop",
            "transformer.ln_f",
            "lm_head",
        ],
    ): 3
}
heterogeneous_templates = {
    PipelineTemplate(
        config,
        ["0"],
        [2, 2, 2],
        [
            [f"transformer.{emb}" for emb in ["wte", "wpe", "drop"]]
            + [f"transformer.h.0"],
            [f"transformer.h.{i}" for i in range(1, 4)],
            ["transformer.ln_f", "lm_head"],
        ],
        [
            "transformer.wte",
            "transformer.wpe",
            "transformer.drop",
            "transformer.ln_f",
            "lm_head",
        ],
    ): 1,
    PipelineTemplate(
        config,
        ["0"],
        [2, 2],
        [
            [f"transformer.{emb}" for emb in ["wte", "wpe", "drop"]],
            [f"transformer.h.{i}" for i in range(0, 4)]
            + ["transformer.ln_f", "lm_head"],
        ],
        [
            "transformer.wte",
            "transformer.wpe",
            "transformer.drop",
            "transformer.ln_f",
            "lm_head",
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
            num_microbatches=[3] * sum(pipeline_templates.values()),
        )
        plugin.set_pipeline_templates(pipeline_templates)

        global config
        model = AutoModelForCausalLM.from_config(config)

        optimizer = CPUAdam(model.parameters())
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, 0, 100)

        model, optimizer, _, _, lr_scheduler = plugin.configure(
            model,
            optimizer,
            criterion=lambda outputs, inputs: outputs.loss,
            lr_scheduler=lr_scheduler,
        )

        # TODO: check whether model is split as pipeline template intended
        pipeline_index = expected_pipeline_index[
            next(
                ranks for ranks in expected_pipeline_index.keys() if self.rank in ranks
            )
        ]
        assert plugin._pipeline_index == pipeline_index

        assert isinstance(model, ModelWrapper)
        assert plugin._pipeline_index_to_pipeline[
            plugin._pipeline_index
        ].verify_all_modules_in_stage(model.module, plugin.stage_manager.stage)


instantiate_parametrized_tests(TestHeterogeneousParallelPluginClass)
