from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable, Mapping

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from cornstarch.models.forward_specs import (
    TransformerForwardSpec,
    run_transformer_forward,
)
from cornstarch.models.layer_compile import RepeatedLayerCompileConfig
from cornstarch.models.layer_offload import RepeatedLayerOffloadConfig
from cornstarch.models.model_base import CornstarchModelBase


class CornstarchEncoder(CornstarchModelBase):
    """Unified Cornstarch representation for every non-language encoder.

    Vision and audio encoders follow the same high-level layout even when their
    input preparation differs: a ``pre_encoder`` section builds hidden states, an
    ``encoder_layers`` ``ModuleList`` owns the repeated transformer blocks, and a
    ``post_encoder`` section performs normalization, pooling, projection, or
    other model-family tail work. Keeping this shape consistent lets Cornstarch
    expose the same materialization and offload controls across modalities.

    Vision and audio are properties of inputs and converter-owned forward specs,
    not different execution structures. All encoder converters therefore return
    this same class. That lets lazy materialization and TP/PP/EP walk one stable
    pre/layers/post representation, just as every language converter returns one
    ``CornstarchLanguageModel``.
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

    def _section_names(self) -> tuple[str, str, str]:
        """Three-section layout: pre-encoder, encoder layers, post-encoder."""
        return ("pre_encoder", "encoder_layers", "post_encoder")


# Compatibility for callers that imported the old implementation-oriented name.
# It is an alias, not a second representation or subclass hierarchy.
CornstarchEncoderBase = CornstarchEncoder
