from __future__ import annotations

import copy

import torch
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperModel

from new_cornstarch.models.audio_encoder import CornstarchAudioEncoder


def convert_whisper_config(
    config: WhisperConfig,
    attn_implementation: str | None = None,
) -> CornstarchAudioEncoder:
    with torch.device("meta"):
        hf_model = WhisperModel(copy.deepcopy(config))
    return CornstarchAudioEncoder(
        hf_model,
        config,
        repeated_layers=hf_model.encoder.layers,
        attn_implementation=attn_implementation,
    )
