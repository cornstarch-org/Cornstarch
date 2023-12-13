import torch.distributed as dist
import gc
import numpy as np
import itertools

from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup
from pipeline_template.pipeline_template import PipelineTemplate

DP_AXIS, PP_AXIS, TP_AXIS = 0, 1, 2


class HeterogeneousProcessGroupMesh(ProcessGroupMesh):
    """A helper class to manage the process group mesh.
    Different from original ColossalAI's ProcessGroupMesh, which holds a rank
    per pipeline stage, this HeterogeneousProcessGroupMesh manages ranks per model layer.

    Due to this difference, production of numbers may not be equal to the world size.

    Example: (3, 6, 2)
                pipeline[0]     pipeline[1]     pipeline[2]
                +---+---+       +---+---+       +----+----+
        [D][0]  | 0 | 1 |       | 6 | 7 |       | 10 | 11 |
                +---+---+       +---+---+       +----+----+
        [D][1]  | 0 | 1 |       | 6 | 7 |       | 10 | 11 |
                +---+---+       +---+---+       +----+----+
        [D][2]  | 2 | 3 |       | 6 | 7 |       | 10 | 11 |
                +---+---+       +---+---+       +----+----+
        [D][3]  | 2 | 3 |       | 8 | 9 |       | 12 | 13 |
                +---+---+       +---+---+       +----+----+
        [D][4]  | 4 | 5 |       | 8 | 9 |       | 12 | 13 |
                +---+---+       +---+---+       +----+----+
        [D][5]  | 4 | 5 |       | 8 | 9 |       | 12 | 13 |
                +---+---+       +---+---+       +----+----+
                6 ranks         4 ranks         4 ranks
                3 pp stages     2 pp stages     2 pp stages

        which is equivalent to the following 3D array:
        [
            [[0, 1],    [0, 1],     [2, 3],     [2, 3],     [4, 5],     [4, 5]],
            [[6, 7],    [6, 7],     [6, 7],     [8, 9],     [8, 9],     [8, 9]],
            [[10, 11],  [10, 11],   [10, 11],   [12, 13],   [12, 13],   [12, 13]]
        ]

        There are three pipelines, each of which has 6, 4, and 4 ranks,
        forming 3, 2, and 2 stages pipelines. Second and third pipelines look identical.

        Each DP table represent a single copy of model replica.
        It has rows as many as the number of model layers (in this example, 6 layers).
        For example, first layer of first data parallel model replica is assigned to rank 0 and 1.
        For each layer, ranks in the same row uses tensor parallelism or ZeRO to hold sharded
        model parameters and train the layer.

        Currently it does not support heterogeneous number of rank assignment to different layer
        (e.g. 4 ranks for 0th layer, 2 ranks for 2nd layer).
        TODO (insujang): implement it
    Args:
        plugin (HeterogeneousParallelPlugin): A plugin with an execution plan of heterogeneous pipelines.

    Attributes:
        shape (tuple[int, ...]): The shape of the process group mesh.
        rank (int): The rank of the current process.
    """

    def __init__(
        self,
        pipeline_templates: dict[PipelineTemplate, int],
        tp_size: int,
    ):
        assert pipeline_templates and all(
            num_template > 0 for num_template in pipeline_templates.values()
        ), "At least one pipeline template must be given."
        assert all(
            pt.num_layers == next(iter(pipeline_templates)).num_layers
            for pt in pipeline_templates
        ), "All pipeline templates must have the same number of layers."

        # Sort pipeline templates in terms of number of nodes used.
        pipeline_templates: list[tuple[PipelineTemplate, int]] = sorted(
            pipeline_templates.items(), key=lambda pt: len(pt[0].node_ids)
        )

        self._shape = (
            sum(num_pt for _, num_pt in pipeline_templates),
            pipeline_templates[0][0].num_layers,
            tp_size,
        )
        self._rank = dist.get_rank()
        self._mesh: np.array = np.empty(self._shape, dtype=object)
        rank = 0
        num_pipelines = 0
        for template, num_template in pipeline_templates:
            for _ in range(num_template):
                pipeline_ranks = np.empty((template.num_layers, tp_size), dtype=object)
                next_layer_index = 0
                for modules in template.modules_per_stage:
                    ranks_per_stage = list(range(rank, rank + tp_size))
                    for _ in range(len(modules)):
                        pipeline_ranks[next_layer_index] = ranks_per_stage
                        next_layer_index += 1
                    rank += tp_size
                self._mesh[num_pipelines] = pipeline_ranks
                num_pipelines += 1
        self._coords = HeterogeneousProcessGroupMesh.unravel(self._rank, self._mesh)
        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}

    def __del__(self):
        for group in self._ranks_to_group.values():
            if group is None:
                continue
            dist.destroy_process_group(group)

    @property
    def coords(self) -> list[tuple[int, ...]]:
        """The process coordinates.

        Returns:
            list[tuple[int, ...]]: The process coordinates.
        """
        return self._coords

    @property
    def mesh(self) -> np.array:
        """The process rank mesh.

        Returns:
            np.array: The process rank mesh.
        """
        return self._mesh

    # Inherit self.shape, self.rank, self.ravel.

    @staticmethod
    def unravel(rank: int, mesh: np.array) -> list[tuple[int, ...]]:
        """Convert a rank to a list of coordinates.

        Unlike colossalai.cluster.process_group_mesh.ProcessGroupMesh.unravel,
        our mesh manages process groups per layer; hence the same rank can exist
        in multiple coordinates.

        Args:
            rank (int): Rank to be converted.
            mesh (tuple[int, ...]): A grid of process ranks.

        Returns:
            list[tuple[int, ...]]: List of coordinates of the rank.
        """
        indices = np.where(mesh == rank)
        return list(zip(*indices))

    def get_group(
        self, ranks_in_group: list[int], backend: str | None = None
    ) -> dist.ProcessGroup:
        """Get the process group with the given ranks. It the process group doesn't exist, it will be created.
        Patch the bug that checks if ranks_in_group in self._group_to_ranks;
        it is supposed to check with self._ranks_to_group.

        Args:
            ranks_in_group (List[int]): Ranks in the process group.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group with the given ranks.
        """
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._ranks_to_group:
            group = dist.new_group(ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return self._ranks_to_group[tuple(ranks_in_group)]

    def create_group_along_axis(
        self, axis: int, indices_at_axis: list[int], backend: str | None = None
    ) -> dist.ProcessGroup:
        """Create all process groups along the given axis, and return the one which the current process belongs to.

        Args:
            axis (int): Axis along which the process groups are created.
            indices_at_axis (Optional[List[int]], optional): Indices at the axis. Defaults to None.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        reduced_shape = list(self._shape)
        # the choices on the axis are reduced to 1, since it's determined by `indices_at_axis`
        reduced_shape[axis] = 1
        target_group = None
        # use Cartesian product to generate all combinations of coordinates
        for base_coord in itertools.product(*[range(s) for s in reduced_shape]):
            coords_in_group = ProcessGroupMesh.get_coords_along_axis(
                base_coord, axis, indices_at_axis
            )
            ranks_in_group = tuple(
                set([self._mesh[coord] for coord in coords_in_group])
            )
            if len(ranks_in_group) == 1:
                continue
            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self,
        axis: int,
        indices_at_axis: list[int] | None = None,
        backend: str | None = None,
    ) -> list[ProcessGroup]:
        """Get the process group along the given axis which the current process belongs to.
        If the process group doesn't exist, it will be created.

        Args:
            axis (int): Axis along which the process groups are created.
            indices_at_axis (Optional[List[int]], optional): Indices at the axis. Defaults to None.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            list[ProcessGroup]: The process group along the given axis which the current process belongs to.
            If there are multiple process groups for this rank due to heterogeneous pipelines,
            return all of them.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))
        # Getting coordinates and getting ranks from the coordinates are not the same.
        # self._coords might have multiple coordinates for the same rank.
        # But regardless of which one is used, ranks in the group should be the same.

        groups: list[dist.ProcessGroup] = []

        for coords in self._coords:
            coords_in_group = self.get_coords_along_axis(coords, axis, indices_at_axis)
            ranks_in_group = tuple(
                set([self._mesh[coord] for coord in coords_in_group])
            )

            if ranks_in_group not in self._ranks_to_group:
                # no need to cache it explicitely, since it will be cached in `create_group_along_axis`
                group = self.create_group_along_axis(
                    axis, indices_at_axis, backend=backend
                )
            else:
                group = self._ranks_to_group[ranks_in_group]
            groups.append(group)

        return groups
