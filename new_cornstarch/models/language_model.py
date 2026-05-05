from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from new_cornstarch.models.model_base import CornstarchModelBase
from new_cornstarch.models.repeated_layer import RepeatedLayerStack


class CornstarchLanguageModel(CornstarchModelBase):
    def __init__(
        self,
        hf_model: nn.Module,
        hf_config: PretrainedConfig,
        repeated_layers: Iterable[nn.Module],
        attn_implementation: str | None = None,
    ):
        super().__init__(hf_model, hf_config, attn_implementation=attn_implementation)
        self.repeated_layers = RepeatedLayerStack(repeated_layers)

    def offload_layers_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        self.repeated_layers.offload_to_cpu(layer_indices)

    def materialize_layers(self, device: str | torch.device) -> None:
        self.repeated_layers.materialize_layers(device)
