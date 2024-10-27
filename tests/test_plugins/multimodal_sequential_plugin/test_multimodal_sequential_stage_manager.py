import pytest
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_sequential_plugin.multimodal_sequential_stage_manager import (
    MultimodalSequentialPipelineStageManager,
)
from cornstarch.plugin.multimodal_sequential_plugin.process_group_mesh import (
    MultimodalSequentialProcessGroupMesh,
)

from .common import (
    encoder1_template,
    encoder3_template,
    llm_template_2stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, encoder_templates, llm_template, expected_prev_next_ranks",
    [
        (
            4,
            {encoder1_template: 1},
            (llm_template_2stages, 1, 1),
            [
                {"prev": [3], "next": [1]},
                {"prev": [0], "next": [2]},
                {"prev": [1], "next": [3]},
                {"prev": [2], "next": [0]},
            ],
        ),
        (
            24,
            {encoder1_template: 2},
            (llm_template_2stages, 4, 1),
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
            12,  # encoders are colocated, thus 2*2 + 2*4 = 12
            {encoder1_template: 2, encoder3_template: 2},
            (llm_template_2stages, 4, 1),
            [
                {"prev": [8, 9], "next": [2]},
                {"prev": [10, 11], "next": [3]},
                {"prev": [0], "next": [4, 5]},
                {"prev": [1], "next": [6, 7]},
                {"prev": [2], "next": [8]},  # rank = 4. llm
                {"prev": [2], "next": [9]},
                {"prev": [3], "next": [10]},
                {"prev": [3], "next": [11]},
                {"prev": [4], "next": [0]},  # rank = 8
                {"prev": [5], "next": [0]},
                {"prev": [6], "next": [1]},
                {"prev": [7], "next": [1]},
            ],
        ),
        (
            40,  # 20 ranks * 2 dp
            {encoder1_template: 2},
            (llm_template_2stages, 4, 2),
            [
                {"prev": [24, 25, 28, 29], "next": [4]},
                {"prev": [26, 27, 30, 31], "next": [5]},
                {"prev": [32, 33, 36, 37], "next": [6]},
                {"prev": [34, 35, 38, 39], "next": [7]},
                {"prev": [0], "next": [8, 9, 12, 13]},  # rank 4
                {"prev": [1], "next": [10, 11, 14, 15]},
                {"prev": [2], "next": [16, 17, 20, 21]},
                {"prev": [3], "next": [18, 19, 22, 23]},
                {"prev": [4], "next": [24]},  # rank 8. llm
                {"prev": [4], "next": [25]},
                {"prev": [5], "next": [26]},
                {"prev": [5], "next": [27]},
                {"prev": [4], "next": [28]},  # rank 12
                {"prev": [4], "next": [29]},
                {"prev": [5], "next": [30]},
                {"prev": [5], "next": [31]},
                {"prev": [6], "next": [32]},  # rank 16
                {"prev": [6], "next": [33]},
                {"prev": [7], "next": [34]},
                {"prev": [7], "next": [35]},
                {"prev": [6], "next": [36]},  # rank 20
                {"prev": [6], "next": [37]},
                {"prev": [7], "next": [38]},
                {"prev": [7], "next": [39]},
                {"prev": [8], "next": [0]},  # rank 24
                {"prev": [9], "next": [0]},
                {"prev": [10], "next": [1]},
                {"prev": [11], "next": [1]},
                {"prev": [12], "next": [0]},  # rank 28
                {"prev": [13], "next": [0]},
                {"prev": [14], "next": [1]},
                {"prev": [15], "next": [1]},
                {"prev": [16], "next": [2]},  # rank 32
                {"prev": [17], "next": [2]},
                {"prev": [18], "next": [3]},
                {"prev": [19], "next": [3]},
                {"prev": [20], "next": [2]},  # rank 36
                {"prev": [21], "next": [2]},
                {"prev": [22], "next": [3]},
                {"prev": [23], "next": [3]},
            ],
        ),
    ],
)
def test_multimodal_sequential_pipeline_stage_manager(
    world_size: int,
    encoder_templates: dict[PipelineTemplate, int],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_prev_next_ranks: list[dict[str, list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultimodalSequentialProcessGroupMesh(encoder_templates, llm_template)
        stage_manager = MultimodalSequentialPipelineStageManager(mesh, mesh.pp_axis)

        assert stage_manager.prev_ranks == expected_prev_next_ranks[rank]["prev"], (
            f"rank {rank} expected to have {expected_prev_next_ranks[rank]['prev']} as previous ranks, "
            f"but got {stage_manager.prev_ranks}."
        )
        assert stage_manager.next_ranks == expected_prev_next_ranks[rank]["next"], (
            f"rank {rank} expected to have {expected_prev_next_ranks[rank]['next']} as next ranks, "
            f"but got {stage_manager.next_ranks}."
        )

        dist.destroy_process_group()
