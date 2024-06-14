from __future__ import annotations

import os
from typing import Union

from transformers.activations import ACT2CLS
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig

VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "dinov2": Dinov2Config,
    "intern_vit_6b": InternVisionConfig,
}


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "multimodal-projector"

    def __init__(
        self,
        encoder_config: PretrainedConfig = None,
        text_config: PretrainedConfig = None,
        projection_type: str = "linear",
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if projection_type not in ["linear", "mlp", "qformer"]:
            raise ValueError(
                f"Unsupported projection type: {projection_type}. "
                f"Supported types are: 'linear', 'mlp', 'qformer'."
            )
        if projection_type != "linear" and activation not in ACT2CLS:
            raise ValueError(
                f"Unsupported activation function: {activation}. "
                f"Supported activations are: {ACT2CLS.keys()}."
            )

        self.projection_type = projection_type
        self.activation = activation

        if encoder_config is not None:
            self.in_features = encoder_config.hidden_size
            self.encoder_model_type = encoder_config.model_type

        if text_config is not None:
            self.out_features = text_config.hidden_size
            self.language_model_type = text_config.model_type
