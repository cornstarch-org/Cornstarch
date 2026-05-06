from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from cornstarch.models import RepeatedLayerCompileConfig, RepeatedLayerOffloadConfig
from cornstarch.models.forward_specs import TransformerForwardSpec, run_transformer_forward


class SyntheticLayer(nn.Module):
    """Small repeated layer with parameter gradients and nonlinear activations."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden_states))


class SyntheticSpec(TransformerForwardSpec):
    """Forward spec that starts from caller-provided hidden states."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        return kwargs["hidden_states"]


class SyntheticModel(nn.Module):
    """Minimal Cornstarch-style model for repeated-layer execution tests."""

    def __init__(
        self,
        layers: nn.ModuleList,
        layer_offload_config: RepeatedLayerOffloadConfig | None = None,
        layer_compile_config: RepeatedLayerCompileConfig | None = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.layer_offload_config = layer_offload_config
        self.layer_compile_config = layer_compile_config or RepeatedLayerCompileConfig(
            enabled=False
        )

    def forward(self, hidden_states: torch.Tensor) -> BaseModelOutput:
        return run_transformer_forward(
            self,
            self.layers,
            SyntheticSpec(),
            {"hidden_states": hidden_states},
        )


def synthetic_layer_stack(width: int = 4, depth: int = 3) -> nn.ModuleList:
    """Create a deterministic stack of synthetic repeated layers."""
    torch.manual_seed(0)
    return nn.ModuleList(SyntheticLayer(width) for _ in range(depth))
