from __future__ import annotations

import inspect
from typing import Any, Callable

import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.dinov2 import Dinov2Config, Dinov2Model

from cornstarch.models.multimodal_language_model import MultimodalLanguageModelConfig


class MultimodalEncoderProjector(nn.Module):
    def __init__(self, encoder: PreTrainedModel, projection: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.config = encoder.config

    def forward(self, *args, **kwargs):
        encoder_outputs = self.encoder(*args, **kwargs)
        return self.projection(encoder_outputs[0])


class MultimodalLanguageModel(PreTrainedModel):
    config_class = MultimodalLanguageModelConfig
    base_model_prefix = "language_model"

    def __init__(
        self,
        config: MultimodalLanguageModelConfig,
        language_model: PreTrainedModel = None,
        encoders: list[MultimodalEncoderProjector] = [],
    ):
        super().__init__(config)

        if language_model is None:
            language_model = AutoModel.from_config(config.text_config)

        if not encoders:
            for idx, encoder_config in enumerate(config.encoder_configs):
                if isinstance(encoder_config, CLIPVisionConfig):
                    encoder = CLIPVisionModel(encoder_config)
                elif isinstance(encoder_config, Dinov2Config):
                    encoder = Dinov2Model(encoder_config)
                else:
                    encoder = AutoModel.from_config(encoder_config)

                if config.projection_type == "linear":
                    projection = nn.Linear(
                        in_features=encoder.config.hidden_size,
                        out_features=language_model.config.hidden_size,
                    )

                encoder_projector = MultimodalEncoderProjector(encoder, projection)
                encoders.append(encoder_projector)

        for idx, encoder in enumerate(encoders):
            self.add_module(f"encoder_{idx}", encoder)

        self.add_module("language_model", language_model)

        self.language_model = language_model
        self.encoders = encoders

    @staticmethod
    def _filter_kwargs(func: Callable, **kwargs) -> dict[str, Any]:
        sig = inspect.signature(func)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    @classmethod
    def from_encoders_llm_pretrained(
        cls,
        encoder_names_or_paths: list[str] = None,
        text_model_name_or_path: str = None,
        projection_type: str = "linear",
        **kwargs,
    ) -> MultimodalLanguageModel:
        r"""
        Example:

        ```python
        >>> # initialize a model from pretrained llama and CLIPVision models.
        >>> model = MultimodalLanguageModel.from_encoders_llm_pretrained(
        ...     encoder_names_or_paths=["openai/clip-vit-base-patch16"],
        ...     text_model_name_or_path="meta-llama/Meta-Llama-3-8B",
        ... )
        ```
        """
        language_model = AutoModelForCausalLM.from_pretrained(
            text_model_name_or_path,
            **MultimodalLanguageModel._filter_kwargs(
                AutoModel.from_pretrained, **kwargs
            ),
        )

        encoder_models: list[MultimodalEncoderProjector] = []
        for encoder_name_or_path in encoder_names_or_paths:
            encoder_config = AutoConfig.from_pretrained(encoder_name_or_path)

            if encoder_config.model_type == "clip":
                encoder_model = CLIPVisionModel.from_pretrained(
                    encoder_name_or_path,
                    **MultimodalLanguageModel._filter_kwargs(
                        CLIPVisionModel.from_pretrained, **kwargs
                    ),
                )
            elif encoder_config.model_type == "dinov2":
                encoder_model = Dinov2Model.from_pretrained(
                    encoder_name_or_path,
                    **MultimodalLanguageModel._filter_kwargs(
                        Dinov2Model.from_pretrained, **kwargs
                    ),
                )
            else:
                encoder_model = AutoModel.from_pretrained(
                    encoder_name_or_path,
                    **MultimodalLanguageModel._filter_kwargs(
                        AutoModel.from_pretrained, **kwargs
                    ),
                )

            # TODO: need to load projection from pretrained as well
            if projection_type == "linear":
                projection = nn.Linear(
                    in_features=encoder_model.config.hidden_size,
                    out_features=language_model.config.hidden_size,
                )

            encoder_models.append(MultimodalEncoderProjector(encoder_model, projection))

        config = MultimodalLanguageModelConfig(
            text_config=language_model.config,
            encoder_configs=[encoder.config for encoder in encoder_models],
            projection_type=projection_type,
            **kwargs,
        )

        model = cls(
            config=config, language_model=language_model, encoders=encoder_models
        )

        return model
