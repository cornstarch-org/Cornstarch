from typing import Optional

import numpy as np
import torch.distributed as dist

from cornstarch.plugin.encoders_replicated_plugin.process_group_mesh import (
    EncodersReplicatedProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class EncodersReplicatedPipelineStageManager(MultiModalPipelineStageManager):
    def __init__(
        self,
        pg_mesh: EncodersReplicatedProcessGroupMesh,
        pipeline_axis: int,
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1

        coords = self.pg_mesh.coords
        prev_coords = []
        next_coords = []

        # Encoders are replicated to all LLM stages.
        # Here we just count the number of stages in the LLM.
        llm_num_stages = pg_mesh.llm_template[0].num_stages

        for i in range(len(coords)):
            if coords[i][self.pipeline_axis] == 0:
                # If this is the first first stage: the last stage is the previous one.
                prev_coords.append(
                    coords[i][: self.pipeline_axis]
                    + (llm_num_stages - 1,)
                    + coords[i][self.pipeline_axis + 1 :]
                )
            else:
                prev_coords.append(
                    coords[i][: self.pipeline_axis]
                    + (coords[i][self.pipeline_axis] - 1,)
                    + coords[i][self.pipeline_axis + 1 :]
                )

            if coords[i][self.pipeline_axis] == llm_num_stages - 1:
                # If this is the last stage: the next stage is the first one.
                next_coords.append(
                    coords[i][: self.pipeline_axis]
                    + (0,)
                    + coords[i][self.pipeline_axis + 1 :]
                )
            else:
                next_coords.append(
                    coords[i][: self.pipeline_axis]
                    + (coords[i][self.pipeline_axis] + 1,)
                    + coords[i][self.pipeline_axis + 1 :]
                )

        self.prev_ranks: list[int] = list(
            sorted(set([self.pg_mesh.mesh[prev_coord] for prev_coord in prev_coords]))
        )
        self.next_ranks: list[int] = list(
            sorted(set([self.pg_mesh.mesh[next_coord] for next_coord in next_coords]))
        )

        self.p2p_group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)[0]

    @property
    def num_stages(self) -> int:
        return self.pg_mesh.llm_template[0].num_stages

    @property
    def stage(self) -> int:
        return self.pg_mesh.coords[0][self.pipeline_axis]

    @property
    def num_stages_in_modal(self) -> int:
        return self.stage

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == 0

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == self.num_stages - 1

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        return self.pg_mesh.llm_template[0].get_num_layers_per_stage()

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        stage = self.stage if stage is None else stage
        num_stages = self.num_stages if num_stages is None else num_stages

        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)
        return (
            num_layers_per_stage_accumulated[stage],
            num_layers_per_stage_accumulated[stage + 1],
        )

    def get_prev_rank(self) -> int:
        return self.prev_ranks[0]

    def get_next_rank(self) -> int:
        return self.next_ranks[0]
