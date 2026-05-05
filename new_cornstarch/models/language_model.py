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
from new_cornstarch.models.layer_offload import RepeatedLayerOffloadConfig
from new_cornstarch.models.model_base import CornstarchModelBase


class CornstarchLanguageModel(CornstarchModelBase):
    """Cornstarch-owned structure for decoder-only language models.

    A converter builds this module by splitting the corresponding Hugging Face
    model into three visible sections: ``pre_decoder`` for embeddings and other
    setup modules, ``decoder_layers`` for the repeated transformer blocks, and
    ``post_decoder`` for normalization, output heads, and similar tail modules.
    The repeated blocks live in an ``nn.ModuleList`` because layer-level
    materialization and offload are core Cornstarch operations rather than hidden
    implementation details.

    The class does not borrow a bound Hugging Face ``forward`` method. Instead,
    it delegates model-family behavior to a ``TransformerForwardSpec`` while the
    shared Cornstarch loop owns iteration through ``decoder_layers``. Checkpoint
    import/export remains Hugging Face-native through the prefix map passed to
    ``CornstarchModelBase``.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        pre_decoder: Mapping[str, nn.Module],
        decoder_layers: Iterable[nn.Module],
        post_decoder: Mapping[str, nn.Module],
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        forward_spec: TransformerForwardSpec,
        attn_implementation: str | None = None,
        layer_offload_config: RepeatedLayerOffloadConfig | None = None,
    ):
        """Register decoder sections, forward spec, and HF state mapping.

        The provided modules are usually created under ``torch.device("meta")``
        by a converter. They should preserve Hugging Face parameter names within
        each section so the prefix map can translate complete state dicts without
        per-parameter special cases.
        """
        super().__init__(
            hf_config,
            hf_to_cornstarch_prefixes=hf_to_cornstarch_prefixes,
            hf_model_factory=hf_model_factory,
            attn_implementation=attn_implementation,
            layer_offload_config=layer_offload_config,
        )
        self.pre_decoder = nn.ModuleDict(pre_decoder)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.post_decoder = nn.ModuleDict(post_decoder)
        self.forward_spec = forward_spec

    def forward(self, **kwargs: Any) -> Any:
        """Run the Cornstarch-owned language-model forward loop."""
        return run_transformer_forward(
            self,
            self.decoder_layers,
            self.forward_spec,
            kwargs,
        )

    def offload_layers_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        """Move selected decoder layers to CPU after they have been materialized."""
        self._offload_module_list_to_cpu(self.decoder_layers, layer_indices)

    def materialize_layers(self, device: str | torch.device) -> None:
        """Allocate or move decoder layers onto the requested device."""
        self._materialize_module_list(self.decoder_layers, torch.device(device))
