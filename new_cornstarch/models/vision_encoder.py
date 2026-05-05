from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Mapping

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from new_cornstarch.models.encoder_base import CornstarchEncoderBase
from new_cornstarch.models.forward_specs import TransformerForwardSpec


class CornstarchVisionEncoder(CornstarchEncoderBase):
    """Cornstarch-owned vision encoder structure, such as CLIP or SigLIP."""

    def __init__(
        self,
        config: PretrainedConfig,
        pre_encoder: Mapping[str, nn.Module],
        encoder_layers: Iterable[nn.Module],
        post_encoder: Mapping[str, nn.Module],
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        forward_spec: TransformerForwardSpec,
        attn_implementation: str | None = None,
    ):
        """Initialize the shared encoder sections with vision-specific modules."""
        super().__init__(
            config,
            pre_encoder=pre_encoder,
            encoder_layers=encoder_layers,
            post_encoder=post_encoder,
            hf_to_cornstarch_prefixes=hf_to_cornstarch_prefixes,
            hf_model_factory=hf_model_factory,
            forward_spec=forward_spec,
            attn_implementation=attn_implementation,
        )
