from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "dinov2": Dinov2Config,
}


class MultimodalLanguageModelConfig(PretrainedConfig):
    r"""
    [`MultimodalLanguageModelConfig] is the configuration class to store the configuration of a
    [`MultimodalLanguageModel]. It is used to instantiate [`MultimodalLanguageModel`] model according to the
    specified arguments, defining the text model and vision model configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import CLIPVisionConfig, LlamaConfig
    >>> from cornstarch.models.multimodal_language_model import MultimodalLanguageModelConfig, MultimodalLanguageModel

    >>> # Initializing a CLIPVisionModel and Llava configuration
    >>> vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
    >>> language_config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")

    >>> # Initializing a MultimodalLanguageModel (with random weights)
    >>> config = MultimodalLanguageModelConfig(text_config=language_config, vision_config=vision_config)
    >>> model = MultimodalLanguageModel(config=config)

    >>> # Initializing a MultimodalLanguageModel from pretrained vision and text models
    >>> model = MultimodalLanguageModel.from_encoders_llm_pretrained("meta-llama/Meta-Llama-3-8B", "openai/clip-vit-base-patch16")

    >>> # Accessing the model configuration
    >>> config_vision = model.config.encoder_configs[0]
    >>> config_text = model.config.text_config
    ```
    """

    model_type = "multimodal_language_model"

    # This config class is compose dof multiple sub-configs
    is_composition = True

    def __init__(
        self,
        text_config: PretrainedConfig,
        vision_config: PretrainedConfig,
        projection_type: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config.pad_token_id is None:
            if text_config.unk_token_id is None:
                raise ValueError(
                    "text_config must have either pad_token_id or unk_token_id set."
                )
            text_config.pad_token_id = text_config.unk_token_id
        self.text_config = text_config
        self.vision_config = vision_config

        vision_model_type = vision_config.model_type
        vision_config_class = VISION_MODEL_CONFIGS.get(vision_model_type, None)
        if vision_config_class is None:
            raise ValueError(f"Unsupported vision config: {vision_model_type}")

        assert projection_type in ["linear"], "Only linear projection is supported."
        self.projection_type = projection_type
