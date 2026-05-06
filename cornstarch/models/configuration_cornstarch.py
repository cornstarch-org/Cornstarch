from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class CornstarchConfig(PretrainedConfig):
    """Minimal serializable wrapper around a source Hugging Face config.

    Most Cornstarch models are constructed from Hugging Face configs and remain
    checkpoint-compatible with the corresponding Hugging Face model family. This
    config stores the original HF config payload, the broad model kind
    (language, vision, audio, or multimodal), and the attention implementation id
    that should be preserved when converters rebuild the Cornstarch module.

    The wrapper is deliberately small. Architectural details continue to live in
    the source HF config so Cornstarch does not fork model-family configuration
    schemas while it experiments with a different module layout and lazy
    lifecycle.
    """

    model_type = "cornstarch"

    def __init__(
        self,
        hf_config: dict[str, Any] | None = None,
        model_kind: str | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ):
        """Store source config metadata needed to recreate Cornstarch modules."""
        super().__init__(**kwargs)
        self.hf_config = hf_config or {}
        self.model_kind = model_kind
        self.attn_implementation = attn_implementation
