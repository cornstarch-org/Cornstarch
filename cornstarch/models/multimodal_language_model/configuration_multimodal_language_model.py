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
    >>> from transformers import CLIPVisionConfig, LlavaConfig
    >>> from cornstarch.models.multimodal_language_model import MultimodalLanguageModelConfig, MultimodalLanguageModel

    >>> # Initializing a CLIPVisionModel and Llava configuration
    >>> config_vision = CLIPVisionConfig()
    >>> config_text = LlavaConfig()

    >>> config = MultimodalLanguageModelConfig(text_config=config_text, encoder_configs=config_vision)

    >>> # Initializing a MultimodalLanguageModel (with random weights)
    >>> model = MultimodalLanguageModel(config=config)

    >>> # Initializing a MultimodalLanguageModel from pretrained vision and text models
    >>> model = MultimodalLanguageModel.from_encoders_llm_pretrained(["openai/clip-vit-base-patch16"], "meta-llama/Meta-Llama-3-8B")

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
        encoder_configs: PretrainedConfig | list[PretrainedConfig],
        projection_type: str = "linear",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_config = AutoConfig.for_model(**text_config.to_dict())

        if not isinstance(encoder_configs, list):
            encoder_configs = [encoder_configs]

        self.encoder_configs: list[PretrainedConfig] = []
        for encoder_config in encoder_configs:
            encoder_model_type = encoder_config.model_type
            vision_config_class = VISION_MODEL_CONFIGS.get(encoder_model_type, None)
            if vision_config_class is not None:
                self.encoder_configs.append(
                    vision_config_class(**encoder_config.to_dict())
                )
            # TODO: add more encoder type
            else:
                self.encoder_configs.append(
                    AutoConfig.for_model(**encoder_config.to_dict())
                )

        assert projection_type in ["linear"], "Only linear projection is supported."
        self.projection_type = projection_type
