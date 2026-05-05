from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_outputs import BaseModelOutput

from new_cornstarch.models.multimodal.configuration import (
    CornstarchEncoderToLanguageProjectorConfig,
)


class CornstarchProjector(nn.Module):
    """Project modality encoder features into a language model hidden size."""

    config: CornstarchEncoderToLanguageProjectorConfig

    def __init__(self, config: CornstarchEncoderToLanguageProjectorConfig):
        super().__init__()
        self.config = config
        if config.projector_type == "linear":
            self.projection = nn.Linear(config.in_features, config.out_features)
        elif config.projector_type == "mlp":
            self.projection = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "in_proj",
                            nn.Linear(config.in_features, config.hidden_features),
                        ),
                        ("activation", get_activation(config.activation)),
                        (
                            "out_proj",
                            nn.Linear(config.hidden_features, config.out_features),
                        ),
                    ]
                )
            )
        elif config.projector_type == "qformer":
            self.projection = CornstarchQFormerProjector(config)
        else:
            raise ValueError(f"Unsupported projector_type: {config.projector_type}")

    def forward(self, hidden_states: torch.Tensor) -> BaseModelOutput:
        """Return projected features in a standard model-output container."""
        projected = self.projection(hidden_states)
        return BaseModelOutput(last_hidden_state=projected)


class CornstarchQFormerProjector(nn.Module):
    """Q-former projector backed by Hugging Face BLIP-2 Q-former modules."""

    def __init__(self, config: CornstarchEncoderToLanguageProjectorConfig):
        super().__init__()
        try:
            from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
            from transformers.models.blip_2.modeling_blip_2 import Blip2QFormerModel
        except ImportError as exc:
            raise ImportError(
                "Q-former projectors require transformers with BLIP-2 Q-former support."
            ) from exc

        qformer_config = Blip2QFormerConfig(
            hidden_size=config.qformer_hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.activation,
            cross_attention_frequency=config.cross_attention_frequency,
            encoder_hidden_size=config.in_features,
        )
        self.query_tokens = nn.Parameter(
            torch.empty(1, config.num_query_tokens, config.qformer_hidden_size)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.qformer = Blip2QFormerModel(qformer_config)
        self.out_proj = nn.Linear(config.qformer_hidden_size, config.out_features)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        query_tokens = self.query_tokens.expand(hidden_states.shape[0], -1, -1)
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=hidden_states,
            return_dict=True,
        )
        return self.out_proj(qformer_outputs.last_hidden_state)
