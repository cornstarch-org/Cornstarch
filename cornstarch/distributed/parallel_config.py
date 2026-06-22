from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cornstarch.distributed.context_parallel.splitters import ContextParallelSplitter


@dataclass
class ParallelConfig:
    """Per-modality parallelization configuration (Option C surface).

    Callers supply one of these to ``ParallelizationPlan.parallelize()`` for
    each module that should be distributed.  Every degree is explicit and
    per-modality, matching the multimodal mental model: a vision encoder can be
    tensor-parallel only while the language model is tensor + pipeline + context
    parallel, exactly like the legacy ``ModalParallelPlugin`` allowed but with no
    per-model policy machinery and no ``PipelineTemplate`` dependency.

    Degrees
    -------
    - ``tensor_parallel_size`` (``tp``): DTensor column/row weight sharding.
    - ``pipeline_parallel_size`` (``pp``): number of pipeline stages.
    - ``context_parallel_size`` (``cp``): sequence is split across these ranks
      (data-side); requires a ``context_parallel_splitter``.
    - ``data_parallel_size`` (``dp``): replicas trained on different data shards.
    - ``expert_parallel_size`` (``ep``): MoE experts sharded across these ranks.

    All process groups and ``DeviceMesh`` handles are constructed internally by
    ``ParallelizationPlan.materialize()`` — callers never create or pass them
    directly.  When ``context_parallel_size > 1`` a splitter must be provided so
    the dataloader knows how to partition sequences across CP ranks.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_splitter: ContextParallelSplitter | None = None

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "context_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if self.context_parallel_size > 1 and self.context_parallel_splitter is None:
            raise ValueError(
                "context_parallel_splitter must be provided when "
                "context_parallel_size > 1."
            )

    @property
    def ranks_per_replica(self) -> int:
        """Ranks one data-parallel replica of this modality consumes."""
        return (
            self.pipeline_parallel_size
            * self.context_parallel_size
            * self.tensor_parallel_size
            * self.expert_parallel_size
        )
