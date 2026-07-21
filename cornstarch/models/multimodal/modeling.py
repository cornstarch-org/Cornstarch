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
        # The projector is the encoder tail. Intermediate PP stages hand their
        # raw activation to the next stage and only the final stage projects.
        pipeline_mesh = getattr(self, "_pipeline_mesh", None)
        if pipeline_mesh is not None and not pipeline_mesh.is_last_stage():
            return encoder_outputs
        hidden_states = _first_output_tensor(encoder_outputs)
        return self.projector(hidden_states)


class CornstarchFusedModalityEncoder(nn.Module):
    """Ordered group of modality encoders sharing one parallel configuration."""

    def __init__(self, encoders: Mapping[str, CornstarchModalityEncoder]):
        super().__init__()
        if not encoders:
            raise ValueError("A fused modality encoder requires at least one child.")
        ordered: dict[str, CornstarchModalityEncoder] = {}
        output_widths: set[int] = set()
        for name, module in encoders.items():
            if not name:
                raise ValueError("Fused modality names must be nonempty.")
            if not isinstance(module, CornstarchModalityEncoder):
                raise TypeError(
                    "Fused children must be CornstarchModalityEncoder instances; "
                    f"got {type(module).__name__} for {name!r}."
                )
            if module.modality is not None and module.modality != name:
                raise ValueError(
                    f"Registry name {name!r} does not match child modality "
                    f"{module.modality!r}."
                )
            module.modality = name
            ordered[name] = module
            output_widths.add(int(module.projector.config.out_features))
        if len(output_widths) != 1:
            raise ValueError(
                "Every fused child projector must target the same language hidden size."
            )
        self.encoders = nn.ModuleDict(ordered)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.encoders.keys())

    @property
    def config(
        self,
    ) -> Mapping[str, tuple[PretrainedConfig, CornstarchEncoderToLanguageProjectorConfig]]:
        return {name: module.config for name, module in self.encoders.items()}

    @property
    def output_hidden_size(self) -> int:
        first = next(iter(self.encoders.values()))
        return int(first.projector.config.out_features)

    def set_empty_init(self) -> None:
        for module in self.encoders.values():
            module.set_empty_init()

    def set_random_init(self) -> None:
        for module in self.encoders.values():
            module.set_random_init()

    def set_checkpoint_init(
        self,
        checkpoints: Mapping[str, Mapping[str, Any]],
    ) -> None:
        unknown = set(checkpoints) - set(self.encoders)
        if unknown:
            raise ValueError(f"Unknown fused checkpoint modalities: {sorted(unknown)}")
        for name, module in self.encoders.items():
            if name in checkpoints:
                module.set_checkpoint_init(**dict(checkpoints[name]))

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> CornstarchFusedModalityEncoder:
        for module in self.encoders.values():
            module.materialize(device, dtype=dtype)
        return self

    def forward(
        self,
        inputs: Mapping[str, Mapping[str, Any]] | None = None,
        **modality_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        provided = dict(inputs or {})
        duplicates = set(provided) & set(modality_inputs)
        if duplicates:
            raise ValueError(f"Duplicate fused modality inputs: {sorted(duplicates)}")
        provided.update(modality_inputs)
        unknown = set(provided) - set(self.encoders)
        if unknown:
            raise ValueError(f"Unknown fused modalities: {sorted(unknown)}")
        return {
            name: module(**dict(provided[name]))
            for name, module in self.encoders.items()
            if name in provided
        }


def build_fused_modality_encoder(
    encoders: Mapping[str, CornstarchModalityEncoder],
) -> CornstarchFusedModalityEncoder:
    """Build one ordered fused producer from already wrapped modality encoders."""
    return CornstarchFusedModalityEncoder(encoders)


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
        if "last_hidden_state" in output:
            return output["last_hidden_state"]
        if "hidden_states" in output:
            return output["hidden_states"]
        raise KeyError("Output mapping has neither last_hidden_state nor hidden_states.")
    if isinstance(output, tuple):
        return output[0]
    raise TypeError(f"Cannot extract hidden states from output of type {type(output).__name__}.")


def _is_meta(module: nn.Module) -> bool:
    tensors = list(module.parameters()) + list(module.buffers())
    return bool(tensors) and any(tensor.is_meta for tensor in tensors)
