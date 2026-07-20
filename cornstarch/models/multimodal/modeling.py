from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from cornstarch.models.encoder_base import CornstarchEncoder
from cornstarch.models.multimodal.configuration import (
    CornstarchEncoderToLanguageProjectorConfig,
)
from cornstarch.models.multimodal.projector import CornstarchProjector


class CornstarchModalityEncoder(nn.Module):
    """Encoder-plus-projector unit for one non-text modality.

    Multimodal Cornstarch plans treat modality execution as a self-contained
    operation: run the modality encoder, extract its hidden states, and project
    those states into the language model hidden size before any text merge
    happens. This module packages that contract so the execution plan does not
    need to know whether the underlying encoder is vision, audio, or another
    future modality.

    The wrapped encoder remains a normal Cornstarch encoder with its own lazy
    materialization and Hugging Face checkpoint behavior. The projector is part
    of the modality unit because feature dimensionality alignment is inseparable
    from modality output semantics. The forward result is already language-sized
    and can be scattered directly into placeholder token positions by
    ``CornstarchExecutionPlan``.
    """

    def __init__(
        self,
        encoder: CornstarchEncoder,
        projector: CornstarchProjector,
        modality: str | None = None,
    ):
        super().__init__()
        if projector.config.in_features != getattr(encoder.config, "hidden_size", None):
            raise ValueError(
                "Projector input size must match encoder hidden size: "
                f"expected {getattr(encoder.config, 'hidden_size', None)}, "
                f"got {projector.config.in_features}."
            )
        self.encoder = encoder
        self.projector = projector
        self.modality = modality

    @property
    def config(self) -> tuple[PretrainedConfig, CornstarchEncoderToLanguageProjectorConfig]:
        """Return the paired encoder and projector configs.

        The tuple mirrors the ownership boundary of this module. There is no
        single required multimodal root config; callers can serialize or inspect
        the modality encoder config and the projection config independently.
        """
        return self.encoder.config, self.projector.config

    def set_empty_init(self) -> None:
        """Configure the wrapped encoder for uninitialized materialization.

        The projector always materializes with ``reset_parameters`` regardless of
        the encoder's init plan; it is a freshly composed bridge module with no
        checkpoint of its own.
        """
        self.encoder.set_empty_init()

    def set_random_init(self) -> None:
        """Configure the wrapped encoder for default random initialization."""
        self.encoder.set_random_init()

    def set_checkpoint_init(
        self,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        checkpoint_path: str | Path | None = None,
        model_name_or_path: str | None = None,
    ) -> None:
        """Configure the wrapped encoder to load checkpoint weights.

        Threads the checkpoint source (staged ``state_dict``, local
        ``checkpoint_path``, or a Hugging Face Hub ``model_name_or_path``) to the
        encoder. The projector has no checkpoint of its own and is always
        randomly initialized at ``materialize()``.
        """
        self.encoder.set_checkpoint_init(
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
        )

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> CornstarchModalityEncoder:
        """Materialize the grouped encoder and projector as one lifecycle unit.

        Both submodules materialize on ``device`` in ``dtype`` so the projector
        follows its encoder automatically — callers never materialize the
        projector separately.
        """
        self.encoder.materialize(device, dtype=dtype)
        self.projector.materialize(device, dtype=dtype)
        return self

    def forward(self, **kwargs: Any) -> Any:
        """Run the encoder and return projected language-sized modality features.

        Encoder outputs may be tensors, Hugging Face model-output objects,
        mappings, or tuples depending on the source model family. The first
        hidden-state tensor is normalized here before projection so downstream
        merge code receives a standard ``BaseModelOutput`` from the projector.
        """
        encoder_outputs = self.encoder(**kwargs)
        hidden_states = _first_output_tensor(encoder_outputs)
        return self.projector(hidden_states)


def build_modality_encoder(
    encoder: CornstarchEncoder,
    language_model: nn.Module,
    modality: str | None = None,
    projector_type: str = "linear",
    **projector_kwargs: Any,
) -> CornstarchModalityEncoder:
    """Compose a modality encoder from still-lazy endpoint modules.

    The source encoder and target language model only need to expose their
    configs; neither has to be materialized. The generated projector is built on
    the ``meta`` device so the whole modality unit preserves Cornstarch's lazy
    construction invariant until ``materialize()`` is called. This is the public
    way to wrap a raw encoder for parallelization — ``parallelize()`` only
    accepts Cornstarch models and modality encoders, never a bare HF encoder
    (which has no projector).
    """
    with torch.device("meta"):
        projector = CornstarchProjector(
            CornstarchEncoderToLanguageProjectorConfig.from_encoder_and_language_configs(
                encoder.config,
                language_model.config,
                projector_type=projector_type,
                **projector_kwargs,
            )
        )
    return CornstarchModalityEncoder(encoder, projector, modality=modality)


def _first_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, Mapping):
        return output["last_hidden_state"]
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Cannot extract hidden states from output of type {type(output).__name__}.")


def _is_meta(module: nn.Module) -> bool:
    tensors = list(module.parameters()) + list(module.buffers())
    return bool(tensors) and any(tensor.is_meta for tensor in tensors)
