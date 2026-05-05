from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Mapping

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from new_cornstarch.models.model_base import CornstarchModelBase
from new_cornstarch.models.repeated_layer import RepeatedLayerStack


class CornstarchEncoderBase(CornstarchModelBase):
    """Shared Cornstarch-owned encoder structure."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        pre_encoder: Mapping[str, nn.Module],
        repeated_layers: Iterable[nn.Module],
        post_encoder: Mapping[str, nn.Module],
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        attn_implementation: str | None = None,
    ):
        """Register shared encoder sections and HF state mapping."""
        super().__init__(
            hf_config,
            hf_to_cornstarch_prefixes=hf_to_cornstarch_prefixes,
            hf_model_factory=hf_model_factory,
            attn_implementation=attn_implementation,
        )
        self.pre_encoder = nn.ModuleDict(pre_encoder)
        self.repeated_layers = RepeatedLayerStack(repeated_layers)
        self.post_encoder = nn.ModuleDict(post_encoder)

    def offload_layers_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        """Move selected encoder layers to CPU after they have been materialized."""
        self.repeated_layers.offload_to_cpu(layer_indices)

    def materialize_layers(self, device: str | torch.device) -> None:
        """Allocate or move encoder layers onto the requested device."""
        self.repeated_layers.materialize_layers(device)
