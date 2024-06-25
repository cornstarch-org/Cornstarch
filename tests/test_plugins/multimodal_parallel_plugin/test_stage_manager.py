import pytest
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)

from .test_modal_process_group_mesh import (
    encoder1_template,
    encoder2_template,
    llm_template_2stages,
    llm_template_4stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, modal_templates, execution_order, expected_prev_next_ranks",
    [
        (
            24,
            {encoder1_template: 2, llm_template_2stages: 4},
            [(encoder1_template, llm_template_2stages)],
            [
                {"prev": [16, 17], "next": [4]},
                {"prev": [18, 19], "next": [5]},
                {"prev": [20, 21], "next": [6]},
                {"prev": [22, 23], "next": [7]},
                {"prev": [0], "next": [8, 9]},  # rank = 4
                {"prev": [1], "next": [10, 11]},
                {"prev": [2], "next": [12, 13]},
                {"prev": [3], "next": [14, 15]},
                {"prev": [4], "next": [16]},  # rank = 8
                {"prev": [4], "next": [17]},
                {"prev": [5], "next": [18]},
                {"prev": [5], "next": [19]},
                {"prev": [6], "next": [20]},  # rank = 12
                {"prev": [6], "next": [21]},
                {"prev": [7], "next": [22]},
                {"prev": [7], "next": [23]},
                {"prev": [8], "next": [0]},  # rank = 16
                {"prev": [9], "next": [0]},
                {"prev": [10], "next": [1]},
                {"prev": [11], "next": [1]},
                {"prev": [12], "next": [2]},  # rank = 20
                {"prev": [13], "next": [2]},
                {"prev": [14], "next": [3]},
                {"prev": [15], "next": [3]},
            ],
        ),
        (
            18,
            {
                encoder1_template: 2,
                encoder2_template: 2,
                llm_template_2stages: 4,
            },
            [
                (encoder1_template, llm_template_2stages),
                (encoder2_template, llm_template_2stages),
            ],
            [
                {"prev": [14, 15], "next": [2]},  # rank = 0. encoder1
                {"prev": [16, 17], "next": [3]},
                {"prev": [0], "next": [10, 11]},  # rank = 2. connected to llm
                {"prev": [1], "next": [12, 13]},
                {"prev": [14, 15], "next": [6]},  # rank = 4. encoder2
                {"prev": [16, 17], "next": [7]},
                {"prev": [4], "next": [8]},
                {"prev": [5], "next": [9]},
                {"prev": [6], "next": [10, 11]},  # rank = 8. connected to llm
                {"prev": [7], "next": [12, 13]},
                {"prev": [2, 8], "next": [14]},
                {"prev": [2, 8], "next": [15]},
                {"prev": [3, 9], "next": [16]},  # rank = 12
                {"prev": [3, 9], "next": [17]},
                {"prev": [10], "next": [0, 4]},
                {"prev": [11], "next": [0, 4]},
                {"prev": [12], "next": [1, 5]},  # rank = 16
                {"prev": [13], "next": [1, 5]},
            ],
        ),
        (
            84,
            {encoder2_template: 4, llm_template_4stages: 4},
            [(encoder2_template, llm_template_4stages)],
            [
                {"prev": [72], "next": [12]},
                {"prev": [73], "next": [13]},
                {"prev": [74], "next": [14]},
                {"prev": [75], "next": [15]},
                {"prev": [76], "next": [16]},
                {"prev": [77], "next": [17]},
                {"prev": [78], "next": [18]},
                {"prev": [79], "next": [19]},
                {"prev": [80], "next": [20]},
                {"prev": [81], "next": [21]},
                {"prev": [82], "next": [22]},
                {"prev": [83], "next": [23]},
            ]
            + [{"prev": [i - 12], "next": [i + 12]} for i in range(12, 72)]
            + [
                {"prev": [60], "next": [0]},
                {"prev": [61], "next": [1]},
                {"prev": [62], "next": [2]},
                {"prev": [63], "next": [3]},
                {"prev": [64], "next": [4]},
                {"prev": [65], "next": [5]},
                {"prev": [66], "next": [6]},
                {"prev": [67], "next": [7]},
                {"prev": [68], "next": [8]},
                {"prev": [69], "next": [9]},
                {"prev": [70], "next": [10]},
                {"prev": [71], "next": [11]},
            ],
        ),
    ],
)
def test_multimodal_pipeline_stage_manager(
    world_size: int,
    modal_templates: dict[PipelineTemplate, int],
    execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)

        assert stage_manager.prev_ranks == expected_prev_next_ranks[rank]["prev"]
        assert stage_manager.next_ranks == expected_prev_next_ranks[rank]["next"]

        dist.destroy_process_group()
