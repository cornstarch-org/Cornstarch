from typing import List
import torch.distributed as dist

from colossalai.pipeline.stage_manager import PipelineStageManager
from torch.distributed import ProcessGroup
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh


class HeterogeneousPipelineStageManager(PipelineStageManager):
    """PipelineStageManager is a helper class to manage pipeline stages.

    The stage manager is only for a single pipeline, which includes this process rank.
    Thus, self.prev_rank, self.next_rank, and self.p2p_groups might be different across
    different processes.

    StageManager is created when HeterogeneousParallelPlugin is configured for boost.
    """

    def __init__(
        self,
        pg_mesh: HeterogeneousProcessGroupMesh,
        pipeline_axis: int,
        is_virtual: bool = False,
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.prev_rank: int | None = None
        self.next_rank: int | None = None
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}

        if is_virtual:
            raise NotImplementedError("Virtual pipeline is not supported.")

        coords = self.pg_mesh.coords
        prev_coord = (
            coords[0][: self.pipeline_axis]
            + (
                (
                    coords[0][self.pipeline_axis] - 1
                    if coords[0][self.pipeline_axis] > 0
                    # the prev rank of rank0 is the last rank
                    else self.pg_mesh.shape[self.pipeline_axis] - 1
                ),
            )
            + coords[0][self.pipeline_axis + 1 :]
        )
        self.prev_rank = self.pg_mesh.mesh[prev_coord]

        next_coord = (
            coords[-1][: self.pipeline_axis]
            + (
                (
                    coords[-1][self.pipeline_axis] + 1
                    if coords[-1][self.pipeline_axis]
                    < self.pg_mesh.shape[self.pipeline_axis] - 1
                    # the next rank of the last rank is rank0
                    else 0
                ),
            )
            + coords[-1][self.pipeline_axis + 1 :]
        )
        self.next_rank = self.pg_mesh.mesh[next_coord]

        # init p2p process groups
        layers = list(range(self.pg_mesh.shape[self.pipeline_axis]))
        for prev, cur in zip(layers[:-1], layers[1:]):
            group = self.pg_mesh.get_group_along_axis(self.pipeline_axis, [prev, cur])
            # If this rank does not belong to the group, group is None
            if group:
                ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
                # This means both layers belong to a single rank without p2p
                if len(ranks_in_group) == 1:
                    continue
                self.p2p_groups[tuple(ranks_in_group)] = group

    @property
    def num_stages(self) -> int:
        pass

    @property
    def stage(self) -> int:
        pass

    def init_process_group_by_stages(self, stages: list[int]) -> dist.ProcessGroup:
        pass

    # Inherited functions from PipelineStageManager
    # is_first_stage()
    # is_last_stage()
    # get_rank()
    # get_prev_rank()
    # get_next_rank()
    # get_p2p_process_group()
