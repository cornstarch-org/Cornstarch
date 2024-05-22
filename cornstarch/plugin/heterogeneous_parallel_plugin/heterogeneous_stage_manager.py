from typing import Optional

import numpy as np
import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager
from torch.distributed.distributed_c10d import GroupMember

from cornstarch.plugin.heterogeneous_parallel_plugin.heterogeneous_process_group_mesh import (
    HeterogeneousProcessGroupMesh,
)


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
        num_layers_per_stage: Optional[list[int]] = None,
    ):
        self.pg_mesh = pg_mesh
        self.pipeline_axis = pipeline_axis
        self.prev_rank: int | None = None
        self.next_rank: int | None = None
        self.p2p_groups: dict[tuple[int, int], dist.ProcessGroup] = {}
        self.is_interleave = False
        self.num_model_chunks = 1
        if num_layers_per_stage is not None:
            assert len(num_layers_per_stage) == self.num_stages
        self.num_layers_per_stage = num_layers_per_stage

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
                assert group == GroupMember.NON_GROUP_MEMBER or isinstance(
                    group, dist.ProcessGroup
                )
                ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
                # This means both layers belong to a single rank without p2p
                if len(ranks_in_group) == 1:
                    continue
                self.p2p_groups[tuple(ranks_in_group)] = group

    @property
    def num_stages(self) -> int:
        group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)
        if group is None:
            # This is one-stage pipeline
            return 1

        return len(self.pg_mesh.get_ranks_in_group(group))

    @property
    def stage(self) -> int:
        group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)

        if group is None:
            # This is one-stage pipeline
            return 0

        ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
        return ranks_in_group.index(self.get_rank())

    def init_process_group_by_layers(self, layers: list[int]) -> dist.ProcessGroup:
        """Get the process group of the given layers.
        This is used to initialize a process group for shared parameters.

        Args:
            layers (list[int]): List of stages.

        Returns:
            ProcessGroup: Process group of the given stages.
        """
        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, layers)

    def init_process_group_by_stages(self, stages: list[int]) -> dist.ProcessGroup:
        """Get the process group of the given stages.
        This is used to initialize a process group for shared parameters.

        As each stage may include several layers but `stages` argument doesn't include
        layer information, we use the index of the first layer in each stage.

        TDOO: deprecate it and use init_process_group_by_layers instead.
              This need to modify ShardFormer policy, as current policy returns stage indices
              as a key.
        """
        group = self.pg_mesh.get_group_along_axis(self.pipeline_axis)
        assert isinstance(group, dist.ProcessGroup)
        ranks_in_group = self.pg_mesh.get_ranks_in_group(group)
        ranks_in_stages = [ranks_in_group[stage] for stage in stages]

        # Get indices of ranks along the pp axis
        indices_of_ranks = [
            list(zip(*np.where(self.pg_mesh.mesh == rank)))[0][self.pipeline_axis]
            for rank in ranks_in_stages
        ]

        return self.pg_mesh.get_group_along_axis(self.pipeline_axis, indices_of_ranks)
