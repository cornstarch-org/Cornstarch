import math
from typing import Callable, Iterator

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


class HeterogeneousBatchSampler(BatchSampler):
    """Sampler that generates a mini-batch of indices.

    Unlike existing BatchSampler that wraps another sampler, this sampler
    works as a standalone sampler.
    A mini-batch of indices is different for each pipeline template, as they have
    different number of microbatches.

    Args:
        dataset: Dataset used for sampling.
        pipeline_index: Index of the pipeline.
        microbatch_size: Size of the microbatch.
        num_microbatches: Number of microbatches for each pipeline.
        shuffle: If True (default), sampler will shuffle the indices.
        seed: random seed used to shuffle the sampler if shuffle-True.
            This number should be identical across all processes
            in the distributed group. Default: 0.
        drop_last: if True, then the sampler will drop the tail of the data
            to make it evenly divisible across the number of replicas.
            If False, the sampler will add extra indices to make the data
            evenly divisible across the replicas. Default: False.

    Example:
        microbatch_size=2, num_microbatches=[2, 3] (calculated global_batch_size=10)

        >>> list(HeterogeneousBatchSampler(range(20), pipeline_index=0)
        [[0, 1], [2, 3], [10, 11], [12, 13]]
        >>> list(HeterogeneousBatchSampler(range(20), pipeline_index=1)
        [[4, 5], [6, 7], [8, 9], [14, 15], [16, 17], [18, 19]]

        where [0, 1], [2, 3] for 0-th pipeline and [4, 5], [6, 7], [8, 9] for 1-th pipeline
        are consumed in a single pipeline iteration.

    """

    def __init__(
        self,
        dataset: Dataset,
        pipeline_index: int,
        microbatch_size: int,
        num_microbatches: list[int],
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if pipeline_index >= len(num_microbatches):
            raise IndexError(
                f"pipeline_index={pipeline_index} is out of range. "
                f"Given number of microbatches={len(num_microbatches)}"
            )

        self.dataset = dataset
        self.microbatch_size = microbatch_size
        self.pipeline_index = pipeline_index
        self.num_microbatches = num_microbatches
        self.epoch = 0

        if not drop_last:
            logger.warning("For simplicity, drop_last is always assumed True.")
        self.drop_last = True

        self.global_batch_size = microbatch_size * sum(num_microbatches)
        self.num_samples = math.floor(len(dataset) / self.global_batch_size)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        for sample_index in range(self.num_samples):
            index = indices[sample_index] * self.global_batch_size

            # batch start indices for the current pipeline within this iteration
            pipeline_batch_offset = sum(self.num_microbatches[: self.pipeline_index])

            # Return all microbatches for the current iteration at once.
            # This is because the current ColossalAI implementation fetches all microbatches.
            batch_start_index = index + self.microbatch_size * pipeline_batch_offset
            yield list(
                range(
                    batch_start_index,
                    batch_start_index
                    + self.microbatch_size * self.num_microbatches[self.pipeline_index],
                )
            )

    def __len__(self) -> int:
        return self.num_samples


class HeterogeneousDataLoader(DataLoader):
    """
    Data loader that provides an iterable over the given dataset
    and heterogeneous batch sampler.

    Unlike existing any dataloaders, HeterogeneousDataLoader has two-stage
    initialization. First, it is initialized with a dataset and other
    arguments. Then, it configures sampler with pipeline_index and num_microbatches
    to be used for the current pipeline.
    This two-stage initialization design enables to initialize HeterogeneousDataLoader
    before torch.distributed and pipelines are configured.
    """

    def __init__(
        self,
        dataset: Dataset,
        global_batch_size: int = 1,
        microbatch_size: int = 1,
        shuffle: bool = True,
        seed: int = 1024,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.microbatch_size = microbatch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.batch_sampler: BatchSampler = None
        self.collate_fn: Callable = kwargs.get("collate_fn", None)

    def __iter__(self) -> _BaseDataLoaderIter:
        if self.batch_sampler is None:
            raise RuntimeError(
                "HeterogeneousDataLoader must finish second-stage initialization "
                "via plugin.configure() before being used."
            )
        return super().__iter__()

    def __len__(self) -> int:
        if self.batch_sampler is None:
            return len(self.dataset) // self.global_batch_size
        else:
            return super().__len__()

    def configure(self, pipeline_index: int, num_microbatches: list[int]):
        assert self.global_batch_size == self.microbatch_size * sum(num_microbatches), (
            f"inconsistent global batch size={self.global_batch_size}, microbatch_size={self.microbatch_size}, "
            f"and num_microbatches={num_microbatches}"
        )
        self._DataLoader__initialized = False
        batch_sampler = HeterogeneousBatchSampler(
            self.dataset,
            pipeline_index,
            self.microbatch_size,
            num_microbatches,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        super().__init__(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            **self.kwargs,
        )
