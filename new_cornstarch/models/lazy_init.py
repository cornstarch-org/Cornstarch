from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass
class InitializationPlan:
    """Describe how a meta-initialized model should become real tensors."""

    mode: str
    state_dict: Mapping[str, torch.Tensor] | None = None
    checkpoint_path: str | Path | None = None

    @classmethod
    def empty(cls) -> InitializationPlan:
        """Create a plan that allocates uninitialized tensors."""
        return cls(mode="empty")

    @classmethod
    def random(cls) -> InitializationPlan:
        """Create a plan that runs the model's default random initialization."""
        return cls(mode="random")

    @classmethod
    def checkpoint(
        cls,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> InitializationPlan:
        """Create a plan that materializes tensors from checkpoint weights."""
        return cls(mode="checkpoint", state_dict=state_dict, checkpoint_path=checkpoint_path)
