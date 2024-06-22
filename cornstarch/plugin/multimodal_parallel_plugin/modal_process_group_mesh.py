import copy
import itertools
from collections import defaultdict, deque

import numpy as np
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh

from cornstarch.pipeline_template import PipelineTemplate


class MultiModalProcessGroupMesh(ProcessGroupMesh):
    """
    Different from the original `ProcessGroupMesh`, which assumes a unimodal model is given,
    `MultimodalProcessGroupMesh` manages graph of models.

    This makes several differences in implementation:
    - First stage rank may be in the middle of the rank array.
    """

    def __init__(
        self,
        modal_templates: dict[PipelineTemplate, int],
        execution_order: list[tuple[PipelineTemplate, PipelineTemplate]],
    ) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        assert len(modal_templates) > 0, "At least one modal is required."
        for from_modal, to_modal in execution_order:
            assert (
                from_modal in modal_templates
            ), f"{from_modal} not in modal_templates."
            assert to_modal in modal_templates, f"{to_modal} not in modal_templates."

        self.execution_order = execution_order
        self.topological_sorted_modals = self.topological_sort(execution_order)
        assert len(modal_templates) == len(self.topological_sorted_modals)

        self.pp_axis, self.dp_axis, self.tp_axis = 0, 1, 2

        assert (
            dist.get_world_size()
            % sum(
                template.num_stages * tp_size
                for template, tp_size in modal_templates.items()
            )
            == 0
        ), "The world size must be divisible by tp_size * pp_size."
        dp_size = dist.get_world_size() // sum(
            template.num_stages * tp_size
            for template, tp_size in modal_templates.items()
        )

        max_tp_size = max(modal_templates.values())
        meshes: list[list[list[int]]] = [[] for _ in range(dp_size)]
        rank_index = 0
        modal_to_ranks: dict[PipelineTemplate, list[int]] = defaultdict(list)

        for modal in self.topological_sorted_modals:
            tp_size = modal_templates[modal]
            for _ in range(modal.num_stages):
                for dp_index in range(dp_size):
                    # create a list of ranks with length `max_tp_size`, where each rank is repeated `max_tp_size // tp_size` times.
                    # Example: [0, 0, 1, 1] for tp_size=2 and max_tp_size=4
                    ranks = [
                        i
                        for i in range(rank_index, rank_index + tp_size)
                        for _ in range(max_tp_size // tp_size)
                    ]
                    rank_index += tp_size

                    modal_to_ranks[modal].extend(ranks)
                    meshes[dp_index].append(ranks)

        self._rank = dist.get_rank()
        self._mesh = np.array(meshes)
        self._shape = self._mesh.shape
        self.modal_to_ranks = {
            modal: list(set(ranks)) for modal, ranks in modal_to_ranks.items()
        }

        self._coords = MultiModalProcessGroupMesh.unravel(self._rank, self._mesh)
        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}

    @property
    def coords(self) -> list[tuple[int, ...]]:
        """The process coordinates.

        Returns:
            list[tuple[int, ...]]: The process coordinates.
        """
        return self._coords

    @property
    def mesh(self) -> dict[PipelineTemplate, np.ndarray]:
        """The process rank mesh.

        Returns:
            np.ndarray: The process rank mesh.
        """
        return self._mesh

    @staticmethod
    def unravel(rank: int, mesh: np.ndarray) -> list[tuple[int, ...]]:
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

    def topological_sort(
        self, execution_order: list[tuple[PipelineTemplate, PipelineTemplate]]
    ) -> list[PipelineTemplate]:
        """
        Topological sort the modal templates based on the execution order.
        """
        graph: dict[PipelineTemplate, list[PipelineTemplate]] = defaultdict(list)
        in_degree: dict[PipelineTemplate, int] = defaultdict(int)
        out_degree: dict[PipelineTemplate, int] = defaultdict(int)

        for from_modal, to_modal in execution_order:
            graph[from_modal].append(to_modal)
            in_degree[to_modal] += 1
            out_degree[from_modal] += 1
            if from_modal not in in_degree:
                in_degree[from_modal] = 0
            if to_modal not in in_degree:
                in_degree[to_modal] = 0

        self.in_degree = copy.deepcopy(in_degree)
        self.out_degree = copy.deepcopy(out_degree)

        # Find all modals with no incoming edges
        zero_in_degree = deque([modal for modal in in_degree if in_degree[modal] == 0])
        topological_order: list[PipelineTemplate] = []

        while zero_in_degree:
            node = zero_in_degree.popleft()
            topological_order.append(node)

            # Reduce the in-degre of each neigher by 1
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        # Check if topological sort is possible
        if len(topological_order) == len(in_degree):
            return topological_order
        else:
            raise ValueError("Topological sort is not possible.")

    def create_group_along_axis(
        self,
        axis: int,
        indices_at_axis: list[int],
        backend: str | None = None,
    ) -> dist.ProcessGroup:
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

            group = self.get_group(ranks_in_group, backend=backend)
            if self._rank in ranks_in_group:
                target_group = group
        return target_group

    def get_group_along_axis(
        self,
        axis: int,
        indices_at_axis: list[int] | None = None,
        backend: str | None = None,
    ) -> dist.ProcessGroup:
        """Get the process group along the given axis which the current process belongs to.
        If the process group doesn't exist, it will be created.

        A rank may exist multiple times in the mesh, but it should belong to the same tp group.
        If axis is DP, it may belong to different dp groups; however, all dp groups must have
        the same set of ranks, thus it should return only one process group.

        Args:
            axis (int): The axis along which the group is created.
            indices_at_axis (list[int], optional): The indices at the axis. Defaults to None.
            backend (str, optional): The backend to create the group. Defaults to None.

        Returns:
            ProcessGroup: The process group along the given axis which the current process belongs to.
        """
        indices_at_axis = indices_at_axis or list(range(self._shape[axis]))

        coords_in_group = self.get_coords_along_axis(
            self._coords[0], axis, indices_at_axis
        )
        ranks_in_group = tuple(set([self._mesh[coord] for coord in coords_in_group]))
        if ranks_in_group not in self._ranks_to_group:
            return self.create_group_along_axis(axis, indices_at_axis, backend=backend)
        return self._ranks_to_group[ranks_in_group]
