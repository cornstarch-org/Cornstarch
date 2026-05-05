from __future__ import annotations

from collections.abc import Iterable

import torch.nn as nn
from transformers import PretrainedConfig

from new_cornstarch.models.encoder_base import CornstarchEncoderBase


class CornstarchVisionEncoder(CornstarchEncoderBase):
    def __init__(
        self,
        hf_model: nn.Module,
        config: PretrainedConfig,
        repeated_layers: Iterable[nn.Module],
        attn_implementation: str | None = None,
    ):
        super().__init__(
            hf_model,
            config,
            repeated_layers=repeated_layers,
            attn_implementation=attn_implementation,
        )
