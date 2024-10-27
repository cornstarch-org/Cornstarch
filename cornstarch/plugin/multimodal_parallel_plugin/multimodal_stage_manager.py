from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
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

        previous_modals = []
        next_modals = []

        if my_modal in pg_mesh.encoder_templates.keys():
            if pg_mesh.llm_template is not None:
                next_modals.append(pg_mesh.llm_template[0])
                previous_modals.append(pg_mesh.llm_template[0])
            else:
                assert (
                    len(pg_mesh.decoder_templates) == 0
                ), "Encoder-decoder model without llm is not supported."

        elif pg_mesh.llm_template is not None and my_modal == pg_mesh.llm_template[0]:
            if (
                len(pg_mesh.encoder_templates) > 0
                and len(pg_mesh.decoder_templates) > 0
            ):
                previous_modals.extend(list(pg_mesh.encoder_templates.keys()))
                next_modals.extend(list(pg_mesh.decoder_templates.keys()))
            elif len(pg_mesh.encoder_templates) > 0:
                assert len(pg_mesh.decoder_templates) == 0
                previous_modals.extend(list(pg_mesh.encoder_templates.keys()))
                next_modals.extend(list(pg_mesh.encoder_templates.keys()))
            elif len(pg_mesh.decoder_templates) > 0:
                assert len(pg_mesh.encoder_templates) == 0
                previous_modals.extend(list(pg_mesh.decoder_templates.keys()))
                next_modals.extend(list(pg_mesh.decoder_templates.keys()))
        elif my_modal in pg_mesh.decoder_templates.keys():
            assert (
                pg_mesh.llm_template is not None
            ), "Decoder model without llm is not supported."
            previous_modals.append(pg_mesh.llm_template[0])
            next_modals.append(pg_mesh.llm_template[0])

        for i in range(len(coords)):
            if (
                # if this stage is the first first stage
                coords[i][self.pipeline_axis]
                == 0
            ) or (
                # if previous stage is in the different modal
                self.stage_index_to_modal[coords[i][self.pipeline_axis] - 1]
                != my_modal
            ):
                last_stage_indices_of_previous_modals = []
                for previous_modal in previous_modals:
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
                self.stage_index_to_modal[coords[i][self.pipeline_axis] + 1]
                != my_modal
            ):
                first_stage_indices_of_next_modals = []
                for next_modal in next_modals:
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

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        """Is the current stage the first stage.

        NOTE:
            - Even if the stage index is not 0, the stage can still be the first stage in MultiModalPipeline.
            - Determining if the stage is the first is done by checking the modal dependency.

        Returns:
            bool: Whether the current stage is the first stage.
        """
        my_modal = self.stage_index_to_modal[self.stage]
        stage_indices_of_modal = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == my_modal
        ]

        if check_only_in_modal:
            # If `check_only_in_modal` is set True, check only if the rank is the first in the modal
            if self.stage == stage_indices_of_modal[0]:
                return True
            else:
                return False
        else:
            # This is the first stage only if it is the first stage of encoders
            # or llm if there is no encoders.
            if (
                (my_modal in self.pg_mesh.encoder_templates.keys())
                or (
                    len(self.pg_mesh.encoder_templates) == 0
                    and my_modal == self.pg_mesh.llm_template[0]
                )
            ) and self.stage == stage_indices_of_modal[0]:
                return True
            else:
                return False

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        """Is the current stage the last stage.

        NOTE:
            - Even if the stage index is not num_stages - 1, the stage can still be the last stage in MultiModalPipeline.
            - Determining if the stage is the last is done by checking the modal dependency.

        Returns:
            bool: Whether the current stage is the last stage.
        """
        my_modal = self.stage_index_to_modal[self.stage]
        stage_indices_of_modal = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == my_modal
        ]

        if check_only_in_modal:
            # If `check_only_in_modal` is set True, check only if the rank is the last in the modal
            if self.stage == stage_indices_of_modal[-1]:
                return True
            else:
                return False
        else:
            # This is the last stage only if it is the last stage of decoders or
            # llm if there is no decoders
            if (
                (my_modal in self.pg_mesh.decoder_templates.keys())
                or (
                    len(self.pg_mesh.decoder_templates) == 0
                    and my_modal == self.pg_mesh.llm_template[0]
                )
            ) and self.stage == stage_indices_of_modal[-1]:
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

    @property
    def num_stages_in_modal(self) -> int:
        return self.stage_index_to_modal[self.stage].num_stages

    @property
    def stage(self) -> int:
        return self.pg_mesh.coords[0][self.pipeline_axis]

    @property
    def stage_in_modal(self) -> int:
        first_stage_index = next(
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == self.stage_index_to_modal[self.stage]
        )
        return self.stage - first_stage_index

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        """
        Distributed layers across stages.

        Returns:
            - list[int]: the number of layers for each stage
        """
        return list(
            itertools.chain.from_iterable(
                modal.get_num_layers_per_stage()
                for modal in self.pg_mesh.topological_sorted_modals
            )
        )

    def _check_my_rank_in_the_stage(self, stage_index: int) -> bool:
        """
        Check if the current rank is in the stage.

        Args:
            stage_index (int): the stage index

        Returns:
            - bool: whether the current rank is in the stage
        """
        ranks_in_modal = next(
            ranks
            for modal, ranks in self.pg_mesh.modal_to_ranks.items()
            if self.stage_index_to_modal[stage_index] == modal
        )
        return dist.get_rank() in ranks_in_modal

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        """
        Get the start index and end index of layers for each stage in the coresponding modal.
        If this rank is not in the modal, return [0, 0].

        Args:
            layers_per_stage (list[int]): number of layers for each stage
            stage (int): the stage index
            num_stages (int): number of stages
            num_model_chunks (int): number of model chunks

        Returns:
            - tuple[int, int]: the start index and end index of this stage
        """
        stage = self.stage if stage is None else stage
        num_stages = self.num_stages if num_stages is None else num_stages

        if not self._check_my_rank_in_the_stage(stage):
            return (0, 0)

        # Find the first stage index of this modal and subtract it from stage
        # to make it zero-based index
        first_stage_index = next(
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == self.stage_index_to_modal[stage]
        )

        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)
        return (
            num_layers_per_stage_accumulated[stage]
            - num_layers_per_stage_accumulated[first_stage_index],
            num_layers_per_stage_accumulated[stage + 1]
            - num_layers_per_stage_accumulated[first_stage_index],
        )
