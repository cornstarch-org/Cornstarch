import copy
from typing import Optional

import numpy as np
import torch.distributed as dist

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.encoders_colocated_plugin.process_group_mesh import (
    EncodersColocatedProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class CoalescedEncoderStageManager(MultiModalPipelineStageManager):
    def __init__(
        self,
        encoder_template: PipelineTemplate,
        pg_mesh: EncodersColocatedProcessGroupMesh,
    ):
        self.encoder_template = encoder_template
        self.pg_mesh = copy.deepcopy(pg_mesh)
        self.pg_mesh._mesh = np.take(
            pg_mesh.mesh,
            indices=range(encoder_template.num_stages),
            axis=pg_mesh.pp_axis,
        )
        self.pipeline_axis = pg_mesh.pp_axis

    @property
    def num_stages(self) -> int:
        return self.encoder_template.num_stages

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:
        return self.encoder_template.get_num_layers_per_stage()

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:
        stage = self.stage if stage is None else stage
        num_stages = self.num_stages if num_stages is None else num_stages

        if stage >= self.encoder_template.num_stages:
            return (0, 0)

        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)
        return (
            num_layers_per_stage_accumulated[stage],
            num_layers_per_stage_accumulated[stage + 1],
        )

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == 0

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        return self.stage == self.encoder_template.num_stages - 1


class EncodersColocatedPipelineStageManager(MultiModalPipelineStageManager):
    def __init__(
        self,
        pg_mesh: EncodersColocatedProcessGroupMesh,
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

        first_encoder_template = next(iter(pg_mesh.encoder_templates.keys()))
        self.stage_index_to_modal = [
            list(self.pg_mesh.encoder_templates.keys())
        ] * first_encoder_template.num_stages + [
            self.pg_mesh.llm_template[0]
        ] * self.pg_mesh.llm_template[
            0
        ].num_stages

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
            # As encoders are always before llm, we can use the pipeline axis to determine the stage.
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

        self.encoder_stage_managers = {
            encoder_template: CoalescedEncoderStageManager(
                encoder_template, self.pg_mesh
            )
            for encoder_template in pg_mesh.encoder_templates.keys()
        }
        self.p2p_group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)[0]

    @property
    def num_stages(self) -> int:
        return len(self.stage_index_to_modal)

    @property
    def num_stages_in_modal(self) -> int:
        modal = self.stage_index_to_modal[self.stage]
        if isinstance(modal, list):
            # coalesced encoders. All encoders have the same number of stages.
            # Just return the first num_stages.
            return modal[0].num_stages
        else:
            assert isinstance(modal, PipelineTemplate)
            return modal.num_stages

    @property
    def stage(self) -> int:
        return self.pg_mesh.coords[0][self.pipeline_axis]

    def is_first_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        if self.stage == 0:
            return True

        stage_indices_of_llm = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal == self.pg_mesh.llm_template[0]
        ]

        if check_only_in_modal:
            # If `check_only_in_modal` is set True,
            # the first stage of llm should also return True
            if self.stage == stage_indices_of_llm[0]:
                return True
            else:
                return False
        else:
            return False

    def is_last_stage(
        self, ignore_chunk: bool = False, check_only_in_modal: bool = True
    ) -> bool:
        first_encoder_template = next(iter(self.pg_mesh.encoder_templates.keys()))
        encoder_num_stages = first_encoder_template.num_stages
        llm_num_stages = self.pg_mesh.llm_template[0].num_stages

        if self.stage == encoder_num_stages + llm_num_stages - 1:
            return True

        stage_indices_of_encoders = [
            index
            for index, modal in enumerate(self.stage_index_to_modal)
            if modal != self.pg_mesh.llm_template[0]
        ]

        if check_only_in_modal:
            # If `check_only_in_modal` is set True,
            # the last stage of encoders should also return True
            if self.stage == stage_indices_of_encoders[-1]:
                return True
            else:
                return False
        else:
            return False

    def distribute_layers(
        self,
        num_layers: Optional[int] = None,
        num_stages: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
    ) -> list[int]:

        first_encoder_template = next(iter(self.pg_mesh.encoder_templates.keys()))
        num_layers_per_encoder_stage = [0] * first_encoder_template.num_stages

        # encoder stages are not managed by this stage manager, thus return 0.
        return (
            num_layers_per_encoder_stage
            + self.pg_mesh.llm_template[0].get_num_layers_per_stage()
        )

    def _check_my_rank_in_the_stage(self, stage_index: int) -> bool:
        return (
            self.stage_index_to_modal[self.stage]
            == self.stage_index_to_modal[stage_index]
        )

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: Optional[int] = None,
        num_model_chunks: Optional[int] = None,
        num_stages: Optional[int] = None,
    ) -> tuple[int, int]:

        stage = self.stage if stage is None else stage
        num_stages = self.num_stages if num_stages is None else num_stages

        if not self._check_my_rank_in_the_stage(stage):
            return (0, 0)

        first_llm_stage_index = next(
            iter(self.pg_mesh.encoder_templates.keys())
        ).num_stages
        first_stage_index = (
            0 if self.stage < first_llm_stage_index else first_llm_stage_index
        )

        num_layers_per_stage_accumulated = np.insert(np.cumsum(layers_per_stage), 0, 0)

        return (
            num_layers_per_stage_accumulated[stage]
            - num_layers_per_stage_accumulated[first_stage_index],
            num_layers_per_stage_accumulated[stage + 1]
            - num_layers_per_stage_accumulated[first_stage_index],
        )

    def get_prev_rank(self) -> int:
        return self.prev_ranks[0]

    def get_next_rank(self) -> int:
        return self.next_ranks[0]
