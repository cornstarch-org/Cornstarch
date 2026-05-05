from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Mapping

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from new_cornstarch.models.encoder_base import CornstarchEncoderBase


class CornstarchAudioEncoder(CornstarchEncoderBase):
    """Cornstarch-owned audio encoder structure, such as Whisper."""

    def __init__(
        self,
        config: PretrainedConfig,
        pre_encoder: Mapping[str, nn.Module],
        repeated_layers: Iterable[nn.Module],
        post_encoder: Mapping[str, nn.Module],
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        forward_impl: Callable[..., object] | None = None,
        attn_implementation: str | None = None,
    ):
        """Initialize the shared encoder sections with audio-specific modules."""
        super().__init__(
            config,
            pre_encoder=pre_encoder,
            repeated_layers=repeated_layers,
            post_encoder=post_encoder,
            hf_to_cornstarch_prefixes=hf_to_cornstarch_prefixes,
            hf_model_factory=hf_model_factory,
            forward_impl=forward_impl,
            attn_implementation=attn_implementation,
        )
