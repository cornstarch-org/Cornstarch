from __future__ import annotations

from typing import Any

from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig


class CornstarchEncoderToLanguageProjectorConfig(PretrainedConfig):
    """Serializable contract for mapping encoder features into language space.

    A modality encoder and a language model are usually built from independent
    Hugging Face configs, so their hidden sizes do not have to match. This config
    records the projection strategy and the source/target widths needed to make
    their tensors compatible. It is intentionally scoped to the projector rather
    than to a whole multimodal wrapper because Cornstarch users compose concrete
    modules through execution plans.

    ``linear`` and ``mlp`` projectors preserve the input sequence length while
    changing feature size. ``qformer`` uses learnable query tokens to produce a
    fixed number of projected tokens that cross-attend to the encoder features.
    The validation here keeps those architectural choices explicit and catches
    missing dimensions before modules are instantiated.
    """

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
        """Build a projector config from encoder and language model configs.

        This is the preferred constructor when both endpoints are available
        because it derives ``in_features`` and ``out_features`` from the models
        being connected. That keeps projection width choices tied to the actual
        composed modules instead of duplicating hidden sizes by hand.
        """
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
    """Serializable metadata for a user-composed multimodal Cornstarch setup.

    Cornstarch multimodal execution is plan-based and does not require a single
    root ``CornstarchMultimodalModel``. This config therefore records the pieces
    needed to describe a composition without claiming ownership of the concrete
    modules: the language config, per-modality encoder configs, projector
    configs, modality placeholder token ids, and optional checkpoint prefixes.

    The config is useful for saving experiment metadata or reconstructing the
    same module graph later, but preprocessing and tokenization remain external
    responsibilities. Placement is still driven at runtime by ``input_ids`` and
    ``modality_token_ids`` in the execution plan.
    """

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
