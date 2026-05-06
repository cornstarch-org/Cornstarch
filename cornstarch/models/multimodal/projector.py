from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers.activations import get_activation
from transformers.modeling_outputs import BaseModelOutput

from cornstarch.models.multimodal.configuration import (
    CornstarchEncoderToLanguageProjectorConfig,
)


class CornstarchProjector(nn.Module):
    """Project encoder hidden states into the language model feature space.

    Modality encoders produce features in their own hidden size, but multimodal
    language models consume embeddings in the language hidden size. This module
    owns that boundary. It supports simple linear projection, a two-layer MLP,
    and a Q-former-backed projection while presenting the same forward contract
    to ``CornstarchModalityEncoder`` and the execution plan.

    The projector config is explicit about input and output widths so shape
    mismatches are caught when composing modules, not during token merging. The
    forward method returns a Hugging Face ``BaseModelOutput`` with
    ``last_hidden_state`` set to the projected features, matching the convention
    used by encoders and keeping downstream code agnostic to the projector type.
    """

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

    def materialize(self, device: str | torch.device = "cuda") -> CornstarchProjector:
        """Allocate meta projector tensors on a device and initialize them.

        Projectors are commonly generated while composing still-meta modality
        and language modules. This mirrors the Cornstarch model lifecycle for
        that smaller owned module: construction can remain allocation-free, and
        ``CornstarchModalityEncoder.materialize()`` later turns the projection
        parameters into regular randomly initialized tensors.
        """
        device = torch.device(device)
        tensors = list(self.parameters()) + list(self.buffers())
        if tensors and any(tensor.is_meta for tensor in tensors):
            self.to_empty(device=device)
            self.reset_parameters()
        else:
            self.to(device)
        return self

    def reset_parameters(self) -> None:
        """Run default initialization for projector submodules."""
        reset_projection = getattr(self.projection, "reset_parameters", None)
        if callable(reset_projection):
            reset_projection()
            return

        for module in self.projection.modules():
            if module is self.projection:
                continue
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

    def forward(self, hidden_states: torch.Tensor) -> BaseModelOutput:
        """Return projected features in a standard model-output container."""
        projected = self.projection(hidden_states)
        return BaseModelOutput(last_hidden_state=projected)


class CornstarchQFormerProjector(nn.Module):
    """Query-token projector for compressing encoder features before fusion.

    The Q-former path uses learnable query tokens that cross-attend to modality
    encoder hidden states, then projects the query outputs into language hidden
    size. It is useful when the modality feature sequence should be summarized
    into a fixed number of language-sized tokens rather than projected
    position-by-position.

    This class intentionally remains an implementation detail of
    ``CornstarchProjector``. Callers select it with
    ``CornstarchEncoderToLanguageProjectorConfig(projector_type="qformer")`` and
    still receive the same projected hidden-state contract as the linear and MLP
    projectors.
    """

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

    def reset_parameters(self) -> None:
        """Initialize query tokens and reset owned projection layers."""
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        reset_parameters = getattr(self.out_proj, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
        post_init = getattr(self.qformer, "post_init", None)
        if callable(post_init):
            post_init()

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
