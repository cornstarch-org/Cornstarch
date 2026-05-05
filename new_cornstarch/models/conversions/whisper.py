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
    """Convert a Whisper config into a meta-initialized Cornstarch audio encoder."""
    with torch.device("meta"):
        hf_model = WhisperModel(copy.deepcopy(config))
    return CornstarchAudioEncoder(
        config,
        pre_encoder={
            "conv1": hf_model.encoder.conv1,
            "conv2": hf_model.encoder.conv2,
            "embed_positions": hf_model.encoder.embed_positions,
        },
        repeated_layers=hf_model.encoder.layers,
        post_encoder={
            "encoder_layer_norm": hf_model.encoder.layer_norm,
            "decoder": hf_model.decoder,
        },
        hf_to_cornstarch_prefixes=(
            ("encoder.conv1.", "pre_encoder.conv1."),
            ("encoder.conv2.", "pre_encoder.conv2."),
            ("encoder.embed_positions.", "pre_encoder.embed_positions."),
            ("encoder.layers.", "repeated_layers.layers."),
            ("encoder.layer_norm.", "post_encoder.encoder_layer_norm."),
            ("decoder.", "post_encoder.decoder."),
        ),
        hf_model_factory=WhisperModel,
        attn_implementation=attn_implementation,
    )
