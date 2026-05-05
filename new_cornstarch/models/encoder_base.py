from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from new_cornstarch.models.forward_specs import (
    TransformerForwardSpec,
    run_transformer_forward,
)
from new_cornstarch.models.model_base import CornstarchModelBase


class CornstarchEncoderBase(CornstarchModelBase):
    """Shared Cornstarch-owned encoder structure."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        pre_encoder: Mapping[str, nn.Module],
        encoder_layers: Iterable[nn.Module],
        post_encoder: Mapping[str, nn.Module],
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        forward_spec: TransformerForwardSpec,
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
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.post_encoder = nn.ModuleDict(post_encoder)
        self.forward_spec = forward_spec

    def forward(self, **kwargs: Any) -> Any:
        """Run the Cornstarch-owned encoder forward loop."""
        return run_transformer_forward(
            self,
            self.encoder_layers,
            self.forward_spec,
            kwargs,
        )

    def offload_layers_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        """Move selected encoder layers to CPU after they have been materialized."""
        self._offload_module_list_to_cpu(self.encoder_layers, layer_indices)

    def materialize_layers(self, device: str | torch.device) -> None:
        """Allocate or move encoder layers onto the requested device."""
        self._materialize_module_list(self.encoder_layers, torch.device(device))
