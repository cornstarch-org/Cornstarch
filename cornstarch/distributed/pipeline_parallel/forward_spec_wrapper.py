"""Stage-aware forward spec wrapper for pipeline parallelism.

``PipelineParallelForwardSpec`` wraps an existing ``TransformerForwardSpec``
so that the shared ``run_transformer_forward`` loop works correctly at each
pipeline stage without model-specific modifications:

- **First stage**: ``embed_inputs()`` runs normally; ``build_output()`` is
  bypassed — only raw hidden states are returned as a dict.
- **Middle stages**: ``embed_inputs()`` extracts hidden states from the
  received dict; ``finalize_hidden_states()`` and ``build_output()`` are
  bypassed.
- **Last stage**: all hooks run normally.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.models.forward_specs import LayerContext, TransformerForwardSpec


class PipelineParallelForwardSpec(TransformerForwardSpec):
    """Wraps a ``TransformerForwardSpec`` for pipeline-stage-aware execution.

    The key that carries hidden states between stages is ``"hidden_states"``.
    Intermediate stages receive a dict ``{"hidden_states": tensor, ...}``
    from ``recv_forward()`` and should pass it as ``input_obj`` into
    ``forward_step()``.  ``embed_inputs()`` on non-first stages extracts the
    hidden state from that dict rather than embedding tokens.
    """

    def __init__(
        self,
        base_spec: TransformerForwardSpec,
        mesh: ModalProcessGroupMesh,
        layer_offset: int = 0,
    ) -> None:
        self._base = base_spec
        self._mesh = mesh
        # PP slices the repeated layers and re-indexes them from 0 on each
        # stage. Specs that key off the *absolute* layer index (e.g. a model
        # with mixed full/linear attention selecting behavior via
        # ``config.layer_types[layer_idx]``) need the global index restored, so
        # the per-layer hooks add this stage's first-layer offset.
        self._layer_offset = layer_offset

    # ------------------------------------------------------------------
    # embed_inputs: first stage embeds tokens; others pass through recv'd h
    # ------------------------------------------------------------------

    def embed_inputs(self, model: nn.Module, **kwargs: Any) -> torch.Tensor:
        if self._mesh.is_first_stage():
            return self._base.embed_inputs(model, **kwargs)
        # Non-first stages receive hidden states from the previous stage.
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError(
                "Non-first pipeline stages expect 'hidden_states' in kwargs "
                "(passed from recv_forward output)."
            )
        return hidden_states

    # ------------------------------------------------------------------
    # Delegate middle hooks to the base spec
    # ------------------------------------------------------------------

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        return self._base.prepare_layer_context(model, hidden_states, **kwargs)

    def should_skip_layer(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> bool:
        return self._base.should_skip_layer(
            model, layer_idx + self._layer_offset, context, **kwargs
        )

    def get_layer_kwargs(
        self, model: nn.Module, layer_idx: int, context: LayerContext, **kwargs: Any
    ) -> dict[str, Any]:
        return self._base.get_layer_kwargs(
            model, layer_idx + self._layer_offset, context, **kwargs
        )

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._base.process_layer_output(
            model, layer_idx + self._layer_offset, layer_output, context, **kwargs
        )

    # ------------------------------------------------------------------
    # finalize_hidden_states: only last stage applies the final norm
    # ------------------------------------------------------------------

    def finalize_hidden_states(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._mesh.is_last_stage():
            return self._base.finalize_hidden_states(model, hidden_states, context, **kwargs)
        return hidden_states

    # ------------------------------------------------------------------
    # build_output: only last stage builds the final output object
    # ------------------------------------------------------------------

    def build_output(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        context: LayerContext,
        **kwargs: Any,
    ) -> Any:
        if self._mesh.is_last_stage():
            return self._base.build_output(model, hidden_states, context, **kwargs)
        # Intermediate stages return hidden states in a dict for P2P transfer.
        return {"hidden_states": hidden_states}
