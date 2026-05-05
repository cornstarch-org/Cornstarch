from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


LayerContext = dict[str, Any]


class TransformerForwardSpec:
    """Model-family hooks for the shared Cornstarch transformer forward loop."""

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        return {}

    def should_skip_layer(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> bool:
        return False

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    def finalize_hidden_states(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> torch.Tensor:
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> Any:
        return BaseModelOutput(last_hidden_state=hidden_states)


def run_transformer_forward(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: TransformerForwardSpec,
    kwargs: dict[str, Any],
) -> Any:
    """Run the shared Cornstarch-owned transformer layer loop."""
    hidden_states = spec.embed_inputs(model, **kwargs)
    loop_kwargs = dict(kwargs)
    loop_kwargs.pop("hidden_states", None)
    context = spec.prepare_layer_context(model, hidden_states, **loop_kwargs)

    for layer_idx, layer in enumerate(layers):
        if spec.should_skip_layer(model, layer_idx, context, **loop_kwargs):
            continue
        layer_output = layer(
            hidden_states,
            **spec.get_layer_kwargs(model, layer_idx, context, **loop_kwargs),
        )
        hidden_states = spec.process_layer_output(
            model, layer_idx, layer_output, context, **loop_kwargs
        )

    hidden_states = spec.finalize_hidden_states(model, hidden_states, context, **loop_kwargs)
    return spec.build_output(model, hidden_states, context, **loop_kwargs)


def _filtered_layer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Forward only generic execution kwargs accepted by HF leaf attention layers."""
    allowed = {
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "is_causal",
        "max_length_q",
        "max_length_k",
        "num_items_in_batch",
    }
    return {key: value for key, value in kwargs.items() if key in allowed}
