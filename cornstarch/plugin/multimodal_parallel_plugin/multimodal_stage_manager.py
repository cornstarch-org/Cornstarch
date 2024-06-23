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
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1
        if num_layers_per_stage is not None:
            assert len(num_layers_per_stage) == self.num_stages
        self.num_layers_per_stage = num_layers_per_stage

        coords = self.pg_mesh.coords
        prev_coords = []
        next_coords = []
        my_modal = self.pg_mesh.stage_index_to_modal[coords[0][self.pipeline_axis]]
        modal_dependency = next(
            md for md in self.pg_mesh.modal_dependencies if md.modal == my_modal
        )
        for i in range(len(coords)):
            if (
                # if this stage is the first first stage
                coords[i][self.pipeline_axis] == 0
            ) or (
                # if previous stage is in the different modal
                self.pg_mesh.stage_index_to_modal[coords[i][self.pipeline_axis] - 1]
                != my_modal
            ):
                last_stage_indices_of_previous_modals = []
                for previous_modal in modal_dependency.previous:
                    last_stage_indices_of_previous_modals.append(
                        [
                            index
                            for index, modal in enumerate(
                                self.pg_mesh.stage_index_to_modal
                            )
                            if modal == previous_modal
                        ][-1]
                    )

                for stage_index in last_stage_indices_of_previous_modals:
                    prev_coords.append(
                        (
                            coords[i][: self.pipeline_axis]
                            + (stage_index,)
                            + coords[i][self.pipeline_axis + 1 :]
                        )
                    )
            else:
                # previous stage is in the same modal
                prev_coords.append(
                    (
                        coords[i][: self.pipeline_axis]
                        + (coords[i][self.pipeline_axis] - 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
                )

            if (
                # if this stage is the last last stage
                coords[i][self.pipeline_axis]
                == self.pg_mesh.shape[self.pipeline_axis] - 1
            ) or (
                # if next stage is in the different modal
                self.pg_mesh.stage_index_to_modal[coords[i][self.pipeline_axis] + 1]
                != my_modal
            ):
                first_stage_indices_of_next_modals = []
                for next_modal in modal_dependency.next:
                    first_stage_indices_of_next_modals.append(
                        [
                            index
                            for index, modal in enumerate(
                                self.pg_mesh.stage_index_to_modal
                            )
                            if modal == next_modal
                        ][0]
                    )

                for stage_index in first_stage_indices_of_next_modals:
                    next_coords.append(
                        (
                            coords[i][: self.pipeline_axis]
                            + (stage_index,)
                            + coords[i][self.pipeline_axis + 1 :]
                        )
                    )
            else:
                # next stage is in the same modal
                next_coords.append(
                    (
                        coords[i][: self.pipeline_axis]
                        + (coords[i][self.pipeline_axis] + 1,)
                        + coords[i][self.pipeline_axis + 1 :]
                    )
                )

        self.prev_ranks: list[int] = list(
            set([self.pg_mesh.mesh[prev_coord] for prev_coord in prev_coords])
        )
        self.next_ranks: list[int] = list(
            set([self.pg_mesh.mesh[next_coord] for next_coord in next_coords])
        )
