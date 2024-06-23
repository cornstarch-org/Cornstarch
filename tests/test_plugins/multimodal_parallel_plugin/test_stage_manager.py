from typing import Any

import pytest
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

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


args = [
    dict(
        world_size=24,
        modal_template={encoder1_template: 2, llm_template_2stages: 4},
        execution_order=[(encoder1_template, llm_template_2stages)],
    ),
    dict(
        world_size=18,
        modal_template={
            encoder1_template: 2,
            encoder2_template: 2,
            llm_template_2stages: 4,
        },
        execution_order=[
            (encoder1_template, llm_template_2stages),
            (encoder2_template, llm_template_2stages),
        ],
    ),
    dict(
        world_size=84,
        modal_template={encoder2_template: 4, llm_template_4stages: 4},
        execution_order=[(encoder2_template, llm_template_4stages)],
    ),
]


@pytest.mark.parametrize("args", args)
def test_multimodal_pipeline_stage_manager(args: dict[str, Any]):
    world_size = args.pop("world_size")
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )
        mesh = MultiModalProcessGroupMesh(**args)
        stage_manager = MultiModalPipelineStageManager(mesh, mesh.pp_axis)
