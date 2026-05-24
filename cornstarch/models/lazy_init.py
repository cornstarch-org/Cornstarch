from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch


_VALID_MODES = frozenset({"empty", "random", "checkpoint"})


@dataclass
class InitializationPlan:
    """Deferred tensor allocation policy for a meta-initialized model.

    New Cornstarch modules are normally constructed on the ``meta`` device so
    converters can assemble very large model topologies without allocating
    parameter storage. An ``InitializationPlan`` records what should happen later
    when ``CornstarchModelBase.materialize()`` is called.

    The plan is intentionally data-only. ``empty`` allocates real tensors without
    filling them, ``random`` asks modules to run their default initialization, and
    ``checkpoint`` assigns weights from an already-loaded state dict or a
    safetensors checkpoint path. Keeping this choice separate from construction
    lets callers stage HF weights before materialization and keeps model classes
    free of ad hoc loading flags.
    """

    mode: str
    state_dict: Mapping[str, torch.Tensor] | None = None
    checkpoint_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown InitializationPlan mode {self.mode!r}. "
                f"Valid modes are: {sorted(_VALID_MODES)}"
            )
        if self.mode == "checkpoint" and self.state_dict is not None and self.checkpoint_path is not None:
            raise ValueError(
                "InitializationPlan.checkpoint() accepts either state_dict or "
                "checkpoint_path, not both."
            )

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
