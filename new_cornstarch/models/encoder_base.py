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
from new_cornstarch.models.layer_compile import RepeatedLayerCompileConfig
from new_cornstarch.models.layer_offload import RepeatedLayerOffloadConfig
from new_cornstarch.models.model_base import CornstarchModelBase


class CornstarchEncoderBase(CornstarchModelBase):
    """Shared structure for Cornstarch-owned non-language encoders.

    Vision and audio encoders follow the same high-level layout even when their
    input preparation differs: a ``pre_encoder`` section builds hidden states, an
    ``encoder_layers`` ``ModuleList`` owns the repeated transformer blocks, and a
    ``post_encoder`` section performs normalization, pooling, projection, or
    other model-family tail work. Keeping this shape consistent lets Cornstarch
    expose the same materialization and offload controls across modalities.

    Subclasses provide the modality label and converters provide the concrete
    Hugging Face leaf modules plus a ``TransformerForwardSpec``. The spec owns
    modality-specific forward details, while this base class preserves the common
    module topology and Hugging Face checkpoint translation inherited from
    ``CornstarchModelBase``.
    """

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
        layer_offload_config: RepeatedLayerOffloadConfig | None = None,
        layer_compile_config: RepeatedLayerCompileConfig | None = None,
    ):
        """Register encoder sections, forward spec, and HF state mapping.

        The section dictionaries are intentionally explicit instead of hiding
        modules behind a root Hugging Face wrapper. Future scheduling and memory
        policies can reason about each repeated block directly through
        ``encoder_layers``.
        """
        super().__init__(
            hf_config,
            hf_to_cornstarch_prefixes=hf_to_cornstarch_prefixes,
            hf_model_factory=hf_model_factory,
            attn_implementation=attn_implementation,
            layer_offload_config=layer_offload_config,
            layer_compile_config=layer_compile_config,
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
        if self.uses_layer_offload:
            assert self.layer_offload_config is not None
            device = self.layer_offload_config.cpu_torch_device
        self._materialize_module_list(self.encoder_layers, torch.device(device))

    def _repeated_layer_module_names(self) -> tuple[str, ...]:
        """Return module names whose tensors are CPU masters under layer offload."""
        return ("encoder_layers",)
