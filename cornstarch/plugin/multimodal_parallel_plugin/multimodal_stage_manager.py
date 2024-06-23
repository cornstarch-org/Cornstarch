from typing import Optional

import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager

from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)


class MultiModalPipelineStageManager(PipelineStageManager):
    """PipelineStageManager is a helper class to manage pipeline stages.

    Unlike traditional unimodal pipeline, where a stage always follows the previous one,
    some stages in multimodal pipeline may be executed in parallel.
    """

    def __init__(
        self,
        pg_mesh: MultiModalProcessGroupMesh,
        pipeline_axis: int,
        num_layers_per_stage: Optional[list[int]] = None,
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.prev_ranks: list[int] = []
        self.next_ranks: list[int] = []
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1
        if num_layers_per_stage is not None:
            assert len(num_layers_per_stage) == self.num_stages
        self.num_layers_per_stage = num_layers_per_stage
