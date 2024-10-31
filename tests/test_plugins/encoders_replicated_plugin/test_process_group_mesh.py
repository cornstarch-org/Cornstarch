import functools
from collections import defaultdict

import pytest
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.encoders_replicated_plugin.process_group_mesh import (
    EncodersReplicatedProcessGroupMesh,
)

from ..common import (
    encoder1_template,
    encoder2_template,
    encoder3_template,
    llm_template_2stages,
    llm_template_4stages,
)


@pytest.fixture(autouse=True)
def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size, llm_template, expected_mesh",
    [
        (
            # 2 stages, 2 replicas
            4,
            (llm_template_2stages, 1, 1),
            [
                [[[0]], [[1]]],
                [[[2]], [[3]]],
            ],
        ),
        (
            # 2 stages, 2 replicas
            16,
            (llm_template_2stages, 2, 2),
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "encoder_templates",
    [
        ([encoder1_template]),
        ([encoder2_template]),
        ([encoder3_template]),
    ],
)
def test_init_process_group_mesh_single_encoder(
    world_size: int,
    encoder_template: list[PipelineTemplate],
    llm_template: tuple[PipelineTemplate, int, int],
    expected_mesh: list[list[list[int]]],
):
    for rank in range(world_size):
        dist.init_process_group(
            backend="fake", store=FakeStore(), rank=rank, world_size=world_size
        )

        mesh = EncodersReplicatedProcessGroupMesh(
            encoder_templates=encoder_template,
            llm_template=llm_template,
        )
        assert (mesh.mesh == expected_mesh).all()

        dist.destroy_process_group()
