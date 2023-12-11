from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

import numpy as np
from pipeline_template.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from pipeline_template.pipeline_template import PipelineTemplate

homogeneous_templates = {
    PipelineTemplate(["0"], [2, 2, 2], [[None], [None, None, None], [None, None]]): 3,
}
heterogeneous_templates = {
    PipelineTemplate(["0"], [2, 2, 2], [[None], [None, None, None], [None, None]]): 1,
    PipelineTemplate(["0"], [2, 2], [[None, None], [None, None, None, None]]): 3,
}


class TestHeterogeneousParallelPluginClass(MultiThreadedTestCase):
    pp_axis = 1

    @property
    def world_size(self):
        return 18

    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @parametrize(
        "pipeline_templates, expected_mesh, expected_num_stages",
        [
            [
                homogeneous_templates,
                [
                    [[0, 1], [2, 3], [2, 3], [2, 3], [4, 5], [4, 5]],
                    [[6, 7], [8, 9], [8, 9], [8, 9], [10, 11], [10, 11]],
                    [[12, 13], [14, 15], [14, 15], [14, 15], [16, 17], [16, 17]],
                ],
                {tuple(list(range(18))): 3},
            ],
            [
                heterogeneous_templates,
                [
                    [[0, 1], [2, 3], [2, 3], [2, 3], [4, 5], [4, 5]],
                    [[6, 7], [6, 7], [8, 9], [8, 9], [8, 9], [8, 9]],
                    [[10, 11], [10, 11], [12, 13], [12, 13], [12, 13], [12, 13]],
                    [[14, 15], [14, 15], [16, 17], [16, 17], [16, 17], [16, 17]],
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


instantiate_parametrized_tests(TestHeterogeneousParallelPluginClass)
