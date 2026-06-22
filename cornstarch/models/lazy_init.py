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
    ``checkpoint`` assigns weights from one of three sources: an already-loaded
    state dict, a local safetensors checkpoint path, or a Hugging Face Hub
    identifier (``model_name_or_path``) whose ``*.safetensors`` shards are
    downloaded and merged at materialization time. Keeping this choice separate
    from construction lets callers stage HF weights before materialization and
    keeps model classes free of ad hoc loading flags.
    """

    mode: str
    state_dict: Mapping[str, torch.Tensor] | None = None
    checkpoint_path: str | Path | None = None
    model_name_or_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unknown InitializationPlan mode {self.mode!r}. "
                f"Valid modes are: {sorted(_VALID_MODES)}"
            )
        if self.mode == "checkpoint":
            sources = sum(
                source is not None
                for source in (self.state_dict, self.checkpoint_path, self.model_name_or_path)
            )
            if sources > 1:
                raise ValueError(
                    "InitializationPlan.checkpoint() accepts at most one of "
                    "state_dict, checkpoint_path, or model_name_or_path."
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
        model_name_or_path: str | None = None,
    ) -> InitializationPlan:
        """Create a plan that materializes tensors from checkpoint weights.

        Exactly one source is used: a pre-loaded ``state_dict``, a local
        ``checkpoint_path`` to a safetensors file, or a Hugging Face Hub
        ``model_name_or_path`` to download and merge.
        """
        return cls(
            mode="checkpoint",
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
        )
