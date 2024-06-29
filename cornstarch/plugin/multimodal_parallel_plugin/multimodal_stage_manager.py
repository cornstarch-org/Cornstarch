from __future__ import annotations

import itertools
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
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1
        self.stage_index_to_modal = list(
            itertools.chain.from_iterable(
                [modal] * modal.num_stages
                for modal in pg_mesh.topological_sorted_modals
            )
        )

        coords = self.pg_mesh.coords
        prev_coords = []
        next_coords = []
        my_modal = self.stage_index_to_modal[coords[0][self.pipeline_axis]]
        modal_dependency = next(
            md for md in self.pg_mesh.modal_dependencies if md.modal == my_modal
        )
        for i in range(len(coords)):
            if (
                # if this stage is the first first stage
                coords[i][self.pipeline_axis] == 0
            ) or (
                # if previous stage is in the different modal
                self.stage_index_to_modal[coords[i][self.pipeline_axis] - 1] != my_modal
            ):
                last_stage_indices_of_previous_modals = []
                for previous_modal in modal_dependency.previous:
                    last_stage_indices_of_previous_modals.append(
                        [
                            index
                            for index, modal in enumerate(self.stage_index_to_modal)
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
                self.stage_index_to_modal[coords[i][self.pipeline_axis] + 1] != my_modal
            ):
                first_stage_indices_of_next_modals = []
                for next_modal in modal_dependency.next:
                    first_stage_indices_of_next_modals.append(
                        [
                            index
                            for index, modal in enumerate(self.stage_index_to_modal)
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
            sorted(set([self.pg_mesh.mesh[prev_coord] for prev_coord in prev_coords]))
        )
        self.next_ranks: list[int] = list(
            sorted(set([self.pg_mesh.mesh[next_coord] for next_coord in next_coords]))
        )

    def is_first_stage(self, ignore_chunk: bool = False) -> bool:
        """Is the current stage the first stage.

        NOTE:
            - Even if the stage index is not 0, the stage can still be the first stage in MultiModalPipeline.
            - Determining if the stage is the first is done by checking the modal dependency.

        Returns:
            bool: Whether the current stage is the first stage.
        """
        coords = self.pg_mesh.coords
        my_modal = self.stage_index_to_modal[coords[0][self.pipeline_axis]]
        modal_dependency = next(
            md for md in self.pg_mesh.modal_dependencies if md.modal == my_modal
        )
        stage_indices_of_modal = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == my_modal
        ]

        # This is the first stage only if in_degree is 0 and this stage is the first one of the modal
        if (
            modal_dependency.in_degree == 0
            and coords[0][self.pipeline_axis] == stage_indices_of_modal[0]
        ):
            return True
        else:
            return False

    def is_last_stage(self, ignore_chunk: bool = False) -> bool:
        """Is the current stage the last stage.

        NOTE:
            - Even if the stage index is not num_stages - 1, the stage can still be the last stage in MultiModalPipeline.
            - Determining if the stage is the last is done by checking the modal dependency.

        Returns:
            bool: Whether the current stage is the last stage.
        """
        coords = self.pg_mesh.coords
        my_modal = self.stage_index_to_modal[coords[0][self.pipeline_axis]]
        modal_dependency = next(
            md for md in self.pg_mesh.modal_dependencies if md.modal == my_modal
        )
        stage_indices_of_modal = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == my_modal
        ]

        # This is the last stage only if out_degree is 0 and this stage is the last one of the modal
        if (
            modal_dependency.out_degree == 0
            and coords[0][self.pipeline_axis] == stage_indices_of_modal[-1]
        ):
            return True
        else:
            return False

    def get_prev_rank(self) -> int:
        raise NotImplementedError(
            "This method is removed from MultimodalPipelineStageManager. "
            "Use `get_prev_ranks` instead."
        )

    def get_next_rank(self) -> int:
        raise NotImplementedError(
            "This method is removed from MultimodalPipelineStageManager. "
            "Use `get_next_ranks` instead."
        )

    def get_prev_ranks(self) -> list[int]:
        return self.prev_ranks

    def get_next_ranks(self) -> list[int]:
        return self.next_ranks

    def init_process_group_by_stages(
        self, stages: list[int]
    ) -> dist.ProcessGroup | list[dist.ProcessGroup]:
        """Get the process group of the given stages.

        Args:
            stages (list[int]): List of stages.

        Returns:
            ProcessGrooup | list[ProcessGroup]: Process groups of the given stages.
            Returns a list only when there are multiple process groups.
        """
        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, stages)

    @property
    def num_stages(self) -> int:
        group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)
        if group is None:
            # This is one-stage pipeline
            return 1

        return self.pg_mesh.shape[self.pipeline_axis]

    def distribute_layers(
        self,
        num_layers: int,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        assert num_layers == sum(
            modal.num_layers for modal in self.pg_mesh.topological_sorted_modals
        ), (
            f"num_layers ({num_layers}) does not match the total number of layers "
            f"({sum(modal.num_layers for modal in self.pg_mesh.topological_sorted_modals)})"
        )

        return list(
            itertools.chain(
                modal.get_num_layers_per_stage()
                for modal in self.pg_mesh.topological_sorted_modals
            )
        )
