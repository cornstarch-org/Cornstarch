from typing import Optional

import torch.distributed as dist

from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)
from cornstarch.plugin.multimodal_sequential_plugin.process_group_mesh import (
    MultimodalSequentialProcessGroupMesh,
)


class MultimodalSequentialPipelineStageManager(MultiModalPipelineStageManager):
    def __init__(
        self,
        pg_mesh: MultimodalSequentialProcessGroupMesh,
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

        self.stage_index_to_modal = [
            list(self.pg_mesh.encoder_templates.keys())
        ] * self.pg_mesh.llm_template[0].num_stages + [
            self.pg_mesh.llm_template[0]
        ] * self.pg_mesh.llm_template[
            0
        ].num_stages

        first_encoder_template = next(iter(pg_mesh.encoder_templates.keys()))
        encoder_num_stages = first_encoder_template.num_stages
        llm_num_stages = self.pg_mesh.llm_template[0].num_stages

        encoder_stage_indices = list(range(encoder_num_stages))
        llm_stage_indices = list(
            range(
                encoder_num_stages,
                encoder_num_stages + self.pg_mesh.llm_template[0].num_stages,
            )
        )

        for i in range(len(coords)):
            if coords[i][self.pipeline_axis] < encoder_num_stages:
                # encoder stages.
                if coords[i][self.pipeline_axis] == 0:
                    # If this is the first first stage: previous stage is llm
                    prev_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (encoder_num_stages + llm_num_stages - 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
                else:
                    prev_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (coords[i][self.pipeline_axis] - 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )

                if coords[i][self.pipeline_axis] == encoder_num_stages - 1:
                    # If this is the last stage: next stage is llm
                    next_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (encoder_num_stages,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
                else:
                    next_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (coords[i][self.pipeline_axis] + 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
            else:
                # llm stages.
                if coords[i][self.pipeline_axis] == encoder_num_stages:
                    # If this is the first stage of llm: previous stage is encoders
                    prev_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (encoder_stage_indices[-1],)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
                else:
                    prev_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (coords[i][self.pipeline_axis] - 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )

                if (
                    coords[i][self.pipeline_axis]
                    == encoder_num_stages + llm_num_stages - 1
                ):
                    # If this is the last last stage: next stage is encoders
                    next_coords.append(
                        coords[i][: self.pipeline_axis]
                        + (encoder_stage_indices[0],)
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

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        raise NotImplementedError

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        raise NotImplementedError

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        raise NotImplementedError

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        raise NotImplementedError
