from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.func import functional_call


@dataclass(frozen=True)
class RepeatedLayerCompileConfig:
    """Configuration for per-layer repeated-block compilation.

    Cornstarch keeps repeated transformer blocks visible so lifecycle features
    can operate at layer granularity. Compilation follows that boundary: each
    repeated layer gets its own cached ``torch.compile`` callable while the
    Python loop, checkpoint boundary, model-family forward spec, and offload
    scheduler remain outside the compiled region.

    Compilation is enabled by default because it is intended to be the normal
    fast path. Use ``enabled=False`` for eager debug runs or when investigating
    compiler-related failures.
    """

    enabled: bool = True
    backend: str = "inductor"
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool | None = None


def layer_compile_enabled(config: RepeatedLayerCompileConfig | None) -> bool:
    """Return whether repeated layer execution should use compiled callables."""
    return config is None or config.enabled


def run_repeated_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, Any],
    config: RepeatedLayerCompileConfig | None,
) -> Any:
    """Run one registered repeated layer through the configured execution path."""
    if not layer_compile_enabled(config):
        return layer(hidden_states, **layer_kwargs)

    compiled_forward = _get_or_create_compiled_forward(layer, config)
    return compiled_forward(hidden_states, **layer_kwargs)


def run_functional_repeated_layer(
    cache_owner: nn.Module,
    template_layer: nn.Module,
    execution_layer: nn.Module,
    hidden_states: torch.Tensor,
    layer_kwargs: dict[str, Any],
    config: RepeatedLayerCompileConfig | None,
) -> Any:
    """Run an offloaded transient layer with a compiled functional call.

    Offload creates short-lived CUDA layer copies whose parameters are views into
    flat transfer tensors. Compiling those transient modules directly would pay
    compilation cost for each prefetch. Instead, this helper caches a compiled
    functional call on the long-lived CPU master layer and supplies the current
    transient layer's tensors as explicit state.
    """
    if not layer_compile_enabled(config):
        return execution_layer(hidden_states, **layer_kwargs)

    template_layer.train(execution_layer.training)
    compiled_forward = _get_or_create_compiled_functional_forward(
        cache_owner,
        template_layer,
        config,
        training=execution_layer.training,
    )
    state = _module_state(execution_layer)
    return compiled_forward(hidden_states, state, layer_kwargs)


def _get_or_create_compiled_forward(
    layer: nn.Module,
    config: RepeatedLayerCompileConfig | None,
) -> Callable[..., Any]:
    cache_key = _compile_cache_key(config)
    cache = layer.__dict__.setdefault("_cornstarch_compiled_layer_forwards", {})
    compiled_forward = cache.get(cache_key)
    if compiled_forward is None:
        compiled_forward = torch.compile(
            layer.forward,
            backend=cache_key[0],
            mode=cache_key[1],
            fullgraph=cache_key[2],
            dynamic=cache_key[3],
        )
        cache[cache_key] = compiled_forward
    return compiled_forward


def _get_or_create_compiled_functional_forward(
    cache_owner: nn.Module,
    template_layer: nn.Module,
    config: RepeatedLayerCompileConfig | None,
    *,
    training: bool,
) -> Callable[[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]], Any]:
    cache_key = (*_compile_cache_key(config), training)
    cache = cache_owner.__dict__.setdefault(
        "_cornstarch_compiled_functional_layer_forwards",
        {},
    )
    compiled_forward = cache.get(cache_key)
    if compiled_forward is None:

        def functional_forward(
            layer_input: torch.Tensor,
            state: dict[str, torch.Tensor],
            kwargs: dict[str, Any],
        ) -> Any:
            return functional_call(template_layer, state, (layer_input,), kwargs)

        compiled_forward = torch.compile(
            functional_forward,
            backend=cache_key[0],
            mode=cache_key[1],
            fullgraph=cache_key[2],
            dynamic=cache_key[3],
        )
        cache[cache_key] = compiled_forward
    return compiled_forward


def _compile_cache_key(
    config: RepeatedLayerCompileConfig | None,
) -> tuple[str, str | None, bool, bool | None]:
    config = config or RepeatedLayerCompileConfig()
    return (config.backend, config.mode, config.fullgraph, config.dynamic)


def _module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    state.update(dict(module.named_parameters(recurse=True)))
    state.update(dict(module.named_buffers(recurse=True)))
    return state
