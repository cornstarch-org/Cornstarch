from torch.testing._internal.common_distributed import MultiThreadedTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from pipeline_template.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousParallelPlugin,
)
from pipeline_template.pipeline_template import PipelineTemplate

homogeneous_templates = {
    PipelineTemplate(["0"], [2, 2, 2], [[None], [None, None, None], [None, None]]): 3,
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
        "pipeline_templates, world_size",
        [[homogeneous_templates, 18]],
        name_fn=lambda pipeline_templates, world_size: "homogeneous"
        if len(pipeline_templates) == 1
        else "heterogeneous",
    )
    def test_plugin_initialize(
        self, pipeline_templates: dict[PipelineTemplate, int], world_size: int
    ):
        plugin = HeterogeneousParallelPlugin(
            tp_size=2,
            microbatch_size=1,
            num_microbatches=[3, 3, 3],
        )
        plugin.set_pipeline_templates(pipeline_templates)


instantiate_parametrized_tests(TestHeterogeneousParallelPluginClass)
