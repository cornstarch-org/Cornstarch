"""Validated explicit contiguous pipeline partition specifications."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelinePartitionSpec:
    """Exclusive layer boundaries, one per pipeline stage."""

    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.boundaries:
            raise ValueError("Pipeline partition boundaries must be nonempty.")
        if any(
            left >= right
            for left, right in zip((0, *self.boundaries[:-1]), self.boundaries)
        ):
            raise ValueError(
                "Pipeline partition boundaries must define ordered nonempty ranges."
            )

    def layer_range(
        self,
        *,
        total_layers: int,
        stage: int,
        num_stages: int,
    ) -> tuple[int, int]:
        if len(self.boundaries) != num_stages:
            raise ValueError(
                f"Expected {num_stages} partition boundaries, got "
                f"{len(self.boundaries)}."
            )
        if self.boundaries[-1] != total_layers:
            raise ValueError(
                f"Partition must be exhaustive: final boundary "
                f"{self.boundaries[-1]} != total_layers {total_layers}."
            )
        if not 0 <= stage < num_stages:
            raise ValueError(f"Invalid pipeline stage {stage} for {num_stages} stages.")
        start = 0 if stage == 0 else self.boundaries[stage - 1]
        return start, self.boundaries[stage]
