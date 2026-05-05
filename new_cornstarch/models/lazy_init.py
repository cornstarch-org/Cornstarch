from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


@dataclass
class InitializationPlan:
    mode: str
    state_dict: Mapping[str, torch.Tensor] | None = None
    checkpoint_path: str | Path | None = None

    @classmethod
    def empty(cls) -> InitializationPlan:
        return cls(mode="empty")

    @classmethod
    def random(cls) -> InitializationPlan:
        return cls(mode="random")

    @classmethod
    def checkpoint(
        cls,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> InitializationPlan:
        return cls(mode="checkpoint", state_dict=state_dict, checkpoint_path=checkpoint_path)
