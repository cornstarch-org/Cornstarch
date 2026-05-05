from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class CornstarchConfig(PretrainedConfig):
    """Serializable config for Cornstarch-wrapped Hugging Face models."""

    model_type = "cornstarch"

    def __init__(
        self,
        hf_config: dict[str, Any] | None = None,
        model_kind: str | None = None,
        attn_implementation: str | None = None,
        **kwargs: Any,
    ):
        """Store the wrapped HF config, model kind, and attention implementation."""
        super().__init__(**kwargs)
        self.hf_config = hf_config or {}
        self.model_kind = model_kind
        self.attn_implementation = attn_implementation
