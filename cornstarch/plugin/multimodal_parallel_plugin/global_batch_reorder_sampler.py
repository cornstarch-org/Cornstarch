from __future__ import annotations

import math
from typing import Callable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


def noop_batch_reorder_fn(indices: List[int]) -> List[int]:
    """No-op implementation of ``batch_reorder_fn``: returns indices unchanged.

    This is the default reorder function used when neither ``metadata_fn`` nor
    ``batch_reorder_fn`` is supplied to ``GlobalBatchReorderSampler``. It also
    serves as a reference for the expected signature.

    A custom ``batch_reorder_fn`` must:

    * Accept a ``List[int]`` of length ``global_batch_size`` — the dataset
      indices that form one global batch, in their current (possibly shuffled)
      order.
    * Return a ``List[int]`` containing exactly the same indices in the desired
      order. Rank 0 will receive the first ``global_batch_size // num_replicas``
      entries, rank 1 the next slice, and so on.

    Example — sort by descending sequence length using a precomputed array::

        lengths: List[int] = [len(dataset[i]["input_ids"]) for i in range(len(dataset))]

        def sort_by_length(indices: List[int]) -> List[int]:
            return sorted(indices, key=lambda i: lengths[i], reverse=True)

    Args:
        indices: Global batch index list of length ``global_batch_size``.

    Returns:
        The same list, unmodified.
    """
    return indices


def constant_cost_metadata_fn(index: int) -> float:
    """No-op implementation of ``metadata_fn``: returns 0.0 for every sample.

    This serves as a reference for the expected signature of ``metadata_fn``.
    Because all samples receive the same cost, no reordering is produced —
    samples remain in their shuffled / sequential order.

    A custom ``metadata_fn`` must:

    * Accept a single ``int`` — a dataset index in ``[0, len(dataset))``.
    * Return a ``float`` representing the computational cost of that sample.
      Higher values mean higher cost; ``GlobalBatchReorderSampler`` sorts each
      global batch in **descending** cost order so that rank 0 always receives
      the most expensive samples.

    The function is called **once per dataset index** during sampler
    construction (``__init__``), not during iteration, so accessing
    ``dataset[index]`` or any pre-computed metadata array is safe and
    will not cause double-loading at training time.

    Example — VLM token-count cost using pre-computed metadata::

        def vlm_cost(index: int) -> float:
            meta = dataset.get_metadata(index)   # lightweight lookup
            return meta["text_length"] + compute_vision_tokens(
                meta["image_height"], meta["image_width"]
            )

    Args:
        index: Dataset index.

    Returns:
        ``0.0`` — constant cost (no reordering effect).
    """
    return 0.0


class GlobalBatchReorderSampler(Sampler[int]):
    """A distributed sampler that reorders samples within each global batch for
    workload balancing across data-parallel ranks.

    Standard ``DistributedSampler`` assigns indices independently to each rank.
    This sampler instead groups indices into global batches of size
    ``global_batch_size``, applies a user-supplied reordering within each
    global batch, and then hands each rank its contiguous slice of
    ``global_batch_size // num_replicas`` indices.

    Because every rank runs identical ``__iter__`` logic with the same seed and
    epoch, the reordering is deterministic and requires no inter-rank
    communication.

    Both ``metadata_fn`` and ``batch_reorder_fn`` are optional. When neither is
    provided the sampler behaves as a standard distributed sampler with
    contiguous per-rank slices and no cost-based reordering.

    Args:
        dataset: The dataset to sample from.
        num_replicas: Number of data-parallel replicas (dp_size).
        rank: Rank of the current process within the data-parallel group.
        global_batch_size: Total number of samples across all replicas in one
            step (``per_replica_batch_size * num_replicas``). Must be divisible
            by ``num_replicas``.
        metadata_fn: A callable ``(index: int) -> float`` that returns a scalar
            cost for sample ``index``. Called for **every** dataset index once
            during ``__init__`` to build a cost array; afterwards only O(1)
            lookups are performed per step. Samples within each global batch are
            sorted in descending cost order so the highest-cost samples go to
            rank 0.  Mutually exclusive with ``batch_reorder_fn``.
        batch_reorder_fn: A callable
            ``(indices: List[int]) -> List[int]`` that receives the global
            batch index list and returns a reordered list. Called once per
            global batch during ``__iter__``. Mutually exclusive with
            ``metadata_fn``.
        shuffle: If ``True``, shuffle the full index list at the start of each
            epoch using ``seed + epoch`` as the generator seed.
        seed: Base random seed used for shuffling.
        drop_last: If ``True``, drop tail samples so the dataset length is
            exactly divisible by the global batch size. If ``False``, pad with
            repeated samples instead.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int,
        rank: int,
        global_batch_size: int,
        metadata_fn: Optional[Callable[[int], float]] = None,
        batch_reorder_fn: Optional[Callable[[List[int]], List[int]]] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if metadata_fn is not None and batch_reorder_fn is not None:
            raise ValueError(
                "Provide either metadata_fn or batch_reorder_fn, not both."
            )

        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas), got rank={rank}, "
                f"num_replicas={num_replicas}."
            )

        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"global_batch_size ({global_batch_size}) must be divisible by "
                f"num_replicas ({num_replicas})."
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.global_batch_size = global_batch_size
        self.per_rank_size = global_batch_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        n = len(dataset)  # type: ignore[arg-type]
        if drop_last:
            num_global_batches = n // self.global_batch_size
        else:
            num_global_batches = math.ceil(n / self.global_batch_size)
        self.total_size = num_global_batches * self.global_batch_size
        self.num_samples = num_global_batches * self.per_rank_size

        if batch_reorder_fn is not None:
            self._reorder_fn: Callable[[List[int]], List[int]] = batch_reorder_fn
        elif metadata_fn is not None:
            # Precompute costs once; sort descending so rank 0 gets highest cost.
            costs = [metadata_fn(i) for i in range(n)]
            self._reorder_fn = lambda indices: sorted(
                indices, key=lambda i: costs[i], reverse=True
            )
        else:
            self._reorder_fn = noop_batch_reorder_fn

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices: List[int] = torch.randperm(
                len(self.dataset), generator=g  # type: ignore[arg-type]
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Pad or trim to an exact multiple of global_batch_size.
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            padding = self.total_size - len(indices)
            if padding > 0:
                indices += indices[:padding]

        # Reorder within each global batch and collect this rank's slice.
        local_indices: List[int] = []
        rank_start = self.rank * self.per_rank_size
        rank_end = rank_start + self.per_rank_size
        for start in range(0, self.total_size, self.global_batch_size):
            global_batch = indices[start : start + self.global_batch_size]
            reordered = self._reorder_fn(global_batch)
            local_indices.extend(reordered[rank_start:rank_end])

        assert len(local_indices) == self.num_samples
        return iter(local_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling. Call at the start of each
        epoch before creating the iterator."""
        self.epoch = epoch
