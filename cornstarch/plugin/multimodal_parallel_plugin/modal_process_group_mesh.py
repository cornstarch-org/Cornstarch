from typing import List

import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh, prod


class ModalProcessGroupMesh(ProcessGroupMesh):
    """
    Different from the original `ProgressGroupMesh`, which includes all ranks,
    `ModalProcessGroupMesh` only includes ranks for a specific modal.
    This makes several differences in implementation.
    TODO: fix it!
    """

    def __init__(self, group: dist.ProcessGroup, *size: int) -> None:
        assert dist.is_initialized(), "Please initialize torch.distributed first."
        world_size = dist.get_world_size(group)
        prod_size = prod(size)
        if world_size != -1:
            assert (
                prod_size == world_size
            ), f"The product of the size({prod_size}) must be equal to the group world size({world_size})."
            self._shape = size
            self._rank = dist.get_rank(group)
            self._coord = ProcessGroupMesh.unravel(self._rank, self._shape)

        self._ranks_to_group: dict[tuple[int, ...], dist.ProcessGroup] = {}
        self._group_to_ranks: dict[dist.ProcessGroup, tuple[int, ...]] = {}
        self._group = group

    def get_group(
        self, ranks_in_group: List[int], backend: str | None = None
    ) -> dist.ProcessGroup:
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._group_to_ranks:
            # Convert local ranks to global ranks
            global_ranks_in_group = [
                dist.get_global_rank(self._group, rank) for rank in ranks_in_group
            ]
            group = dist.new_group(global_ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return super().get_group(ranks_in_group, backend)
