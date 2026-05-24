from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput

from cornstarch.models.layer_compile import run_repeated_layer
from cornstarch.models.layer_offload import (
    activation_checkpoint_recompute_active,
    create_repeated_layer_offload_runtime,
    layer_offload_enabled,
    run_repeated_layers_with_offload,
)


LayerContext = dict[str, Any]


class TransformerForwardSpec:
    """Model-family hooks for Cornstarch's shared transformer loop.

    Cornstarch exposes the repeated transformer blocks as an ``nn.ModuleList`` so
    layers can be materialized, offloaded, or scheduled independently. The common
    loop in ``run_transformer_forward`` owns iteration over that list, while a
    forward spec supplies the family-specific details that normally live inside a
    Hugging Face model ``forward`` method: how inputs become hidden states, what
    masks or position data each layer needs, how tuple outputs are interpreted,
    and how the final model output object is assembled.

    Specs should stay small and declarative. They may call Hugging Face helper
    functions or leaf modules that converters already placed in ``pre_*`` and
    ``post_*`` sections, but they should not hold a bound Hugging Face root model
    or hide extra module ownership. This keeps native Cornstarch forwards
    inspectable and leaves the model structure available for lazy initialization
    and future distributed execution.

    The base implementation describes a plain encoder-style pass. Subclasses
    override only the hooks required by their model family, such as causal mask
    construction for decoder-only language models or pooled-output assembly for
    vision encoders.
    """

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
    """Run repeated transformer layers under a model-family forward spec.

    The loop has one responsibility: execute the Cornstarch-owned layer list in
    order and delegate model-family choices to ``spec``. It deliberately ignores
    optional Hugging Face capture flags such as attentions or hidden-state traces
    unless a spec chooses to implement them, keeping the training path simple and
    predictable.
    """
    layer_offload_config = getattr(model, "layer_offload_config", None)
    hidden_states = spec.embed_inputs(model, **kwargs)
    loop_kwargs = dict(kwargs)
    loop_kwargs.pop("hidden_states", None)
    context = spec.prepare_layer_context(model, hidden_states, **loop_kwargs)

    if layer_offload_enabled(layer_offload_config):
        manager = create_repeated_layer_offload_runtime(layers, layer_offload_config)
        try:
            if len(layers) > 0:
                manager.prefetch(0, direction="forward")
            hidden_states = run_repeated_layers_with_offload(
                model,
                layers,
                spec,
                hidden_states,
                context,
                loop_kwargs,
                layer_offload_config,
                manager=manager,
            )
        except BaseException:
            # Always release transient CUDA copies on failure, even during
            # activation-checkpoint recompute where normal cleanup is deferred.
            manager.free_all()
            raise
        else:
            if not activation_checkpoint_recompute_active():
                manager.free_all()
    else:
        hidden_states = _run_direct_layers(
            model,
            layers,
            spec,
            hidden_states,
            context,
            loop_kwargs,
        )

    hidden_states = spec.finalize_hidden_states(model, hidden_states, context, **loop_kwargs)
    return spec.build_output(model, hidden_states, context, **loop_kwargs)


def _run_direct_layers(
    model: nn.Module,
    layers: nn.ModuleList,
    spec: TransformerForwardSpec,
    hidden_states: torch.Tensor,
    context: LayerContext,
    loop_kwargs: dict[str, Any],
) -> torch.Tensor:
    """Run non-offloaded layers with a checkpoint boundary per layer."""
    for layer_idx, layer in enumerate(layers):
        if spec.should_skip_layer(model, layer_idx, context, **loop_kwargs):
            continue

        def run_layer(
            layer_input: torch.Tensor,
            *,
            layer_idx: int = layer_idx,
            layer: nn.Module = layer,
        ) -> torch.Tensor:
            layer_output = run_repeated_layer(
                layer,
                layer_input,
                spec.get_layer_kwargs(model, layer_idx, context, **loop_kwargs),
                getattr(model, "layer_compile_config", None),
            )
            return spec.process_layer_output(
                model, layer_idx, layer_output, context, **loop_kwargs
            )

        hidden_states = _run_checkpointed_layer(
            model,
            hidden_states,
            run_layer,
        )

    return hidden_states


def _run_checkpointed_layer(
    model: nn.Module,
    hidden_states: torch.Tensor,
    run_layer: Any,
) -> torch.Tensor:
    """Run one repeated layer under mandatory non-reentrant checkpointing."""
    if _should_checkpoint_repeated_layers(model, hidden_states):
        return checkpoint(
            run_layer,
            hidden_states,
            use_reentrant=False,
        )
    return run_layer(hidden_states)


def _should_checkpoint_repeated_layers(
    model: nn.Module, hidden_states: torch.Tensor
) -> bool:
    """Return whether this training pass should checkpoint repeated layers."""
    return (
        model.training
        and torch.is_grad_enabled()
        and not activation_checkpoint_recompute_active()
        and hidden_states.requires_grad
    )


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
