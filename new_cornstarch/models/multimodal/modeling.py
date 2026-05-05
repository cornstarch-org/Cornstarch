from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from new_cornstarch.models.encoder_base import CornstarchEncoderBase
from new_cornstarch.models.multimodal.configuration import (
    CornstarchEncoderToLanguageProjectorConfig,
)
from new_cornstarch.models.multimodal.projector import CornstarchProjector


class CornstarchModalityEncoder(nn.Module):
    """Pair a Cornstarch encoder with the projector used for multimodal fusion."""

    def __init__(
        self,
        encoder: CornstarchEncoderBase,
        projector: CornstarchProjector,
        modality: str | None = None,
    ):
        super().__init__()
        if projector.config.in_features != getattr(encoder.config, "hidden_size", None):
            raise ValueError(
                "Projector input size must match encoder hidden size: "
                f"expected {getattr(encoder.config, 'hidden_size', None)}, "
                f"got {projector.config.in_features}."
            )
        self.encoder = encoder
        self.projector = projector
        self.modality = modality

    @property
    def config(self) -> tuple[PretrainedConfig, CornstarchEncoderToLanguageProjectorConfig]:
        """Return encoder and projector configs."""
        return self.encoder.config, self.projector.config

    def forward(self, **kwargs: Any) -> Any:
        """Run the encoder and return projected language-sized modality features."""
        encoder_outputs = self.encoder(**kwargs)
        hidden_states = _first_output_tensor(encoder_outputs)
        return self.projector(hidden_states)


def _first_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, Mapping):
        return output["last_hidden_state"]
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Cannot extract hidden states from output of type {type(output).__name__}.")


def _is_meta(module: nn.Module) -> bool:
    tensors = list(module.parameters()) + list(module.buffers())
    return bool(tensors) and any(tensor.is_meta for tensor in tensors)
