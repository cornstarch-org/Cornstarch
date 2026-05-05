from __future__ import annotations

from typing import Any

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig


class CornstarchEncoderToLanguageProjectorConfig(PretrainedConfig):
    """Configuration for projecting modality encoder features into LLM hidden space."""

    model_type = "cornstarch_projector"

    def __init__(
        self,
        projector_type: str = "linear",
        in_features: int | None = None,
        out_features: int | None = None,
        hidden_features: int | None = None,
        activation: str = "gelu",
        num_query_tokens: int = 32,
        qformer_hidden_size: int | None = None,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int | None = None,
        cross_attention_frequency: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if projector_type not in {"linear", "mlp", "qformer"}:
            raise ValueError(
                f"Unsupported projector_type: {projector_type}. "
                "Supported types are: 'linear', 'mlp', and 'qformer'."
            )
        if projector_type in {"mlp", "qformer"} and activation not in ACT2FN:
            raise ValueError(
                f"Unsupported activation function: {activation}. "
                f"Supported activations are: {sorted(ACT2FN.keys())}."
            )
        if in_features is None or out_features is None:
            raise ValueError("in_features and out_features must be specified.")

        self.projector_type = projector_type
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features or out_features)
        self.activation = activation
        self.num_query_tokens = int(num_query_tokens)
        self.qformer_hidden_size = int(qformer_hidden_size or self.hidden_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size or self.qformer_hidden_size * 4)
        self.cross_attention_frequency = int(cross_attention_frequency)

    @classmethod
    def from_encoder_and_language_configs(
        cls,
        encoder_config: PretrainedConfig,
        language_config: PretrainedConfig,
        **kwargs: Any,
    ) -> CornstarchEncoderToLanguageProjectorConfig:
        """Build a projector config from an encoder config and language-model config."""
        return cls(
            in_features=_hidden_size_from_config(encoder_config, "encoder_config"),
            out_features=_hidden_size_from_config(language_config, "language_config"),
            **kwargs,
        )


def _hidden_size_from_config(config: PretrainedConfig, config_name: str) -> int:
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "d_model", None)
    if hidden_size is None:
        raise ValueError(
            f"{config_name} must define hidden_size or d_model to build a projector config."
        )
    return int(hidden_size)


class CornstarchMultimodalConfig(PretrainedConfig):
    """Serializable description of a user-composed Cornstarch multimodal model."""

    model_type = "cornstarch_multimodal"

    def __init__(
        self,
        language_config: dict[str, Any] | None = None,
        modality_configs: dict[str, dict[str, Any]] | None = None,
        projector_configs: dict[str, dict[str, Any]] | None = None,
        modality_token_ids: dict[str, int] | None = None,
        checkpoint_prefixes: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_kind = "multimodal"
        self.language_config = language_config or {}
        self.modality_configs = modality_configs or {}
        self.projector_configs = projector_configs or {}
        self.modality_token_ids = modality_token_ids or {}
        self.checkpoint_prefixes = checkpoint_prefixes or {}
