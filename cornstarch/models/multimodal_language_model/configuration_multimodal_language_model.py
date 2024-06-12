from transformers.activations import ACT2CLS
from transformers.configuration_utils import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig

VISION_MODEL_CONFIGS = {
    "clip_vision_model": CLIPVisionConfig,
    "dinov2": Dinov2Config,
    "intern_vit_6b": InternVisionConfig,
}


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "multimodal-projector"

    def __init__(
        self,
        encoder_config: PretrainedConfig,
        text_config: PretrainedConfig,
        projection_type: str = "linear",
        activation: str = "gelu",
    ):
        super().__init__()

        if projection_type not in ["linear", "mlp", "qformer"]:
            raise ValueError(
                f"Unsupported projection type: {projection_type}. "
                f"Supported types are: 'linear', 'mlp', 'qformer'."
            )
        if projection_type != "linear" and activation not in ACT2CLS:
            raise ValueError(
                f"Unsupported activation function: {activation}. "
                f"Supported activations are: {ACT2CLS.keys()}."
            )

        self.projection_type = projection_type
        self.activation = ACT2CLS[activation]

        self.in_features = encoder_config.hidden_size
        self.out_features = text_config.hidden_size

        self.encoder_model_type = encoder_config.model_type
        self.language_model_type = text_config.model_type


# class MultimodalLanguageModelConfig(PretrainedConfig):
#     r"""
#     [`MultimodalLanguageModelConfig] is the configuration class to store the configuration of a
#     [`MultimodalLanguageModel]. It is used to instantiate [`MultimodalLanguageModel`] model according to the
#     specified arguments, defining the text model and vision model configs.

#     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
#     Read the documentation from [`PretrainedConfig`] for more information.

#     Args:
#         text_config (`PretrainedConfig`):
#             HuggingFace configuration of the text model.
#         vision_config (`PretrainedConfig`):
#             HuggingFace configuration of the vision model. Note that only vision configurations are supported,
#             such as CLIPVisionConfig or Dinov2Config.
#         projection_type (`str`, optional, defaults to "linear"): Type of projection from the vision encoder to the LLM.
#             Possible values are: "linear", "mlp", and "qformer".
#         activation (`str`, optional, defaults to "gelu"): Type of activation function to use in the projection layer.
#             Not used when `projection_type` is `linear`. Refer to ACT2CLS in transformers/activations.py for choices:
#             https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
#             Validity is checked when the model is initialized, not when the configuration is created.
#     Examples:

#     ```python
#     >>> from transformers import CLIPVisionConfig, LlamaConfig
#     >>> from cornstarch.models.multimodal_language_model import MultimodalLanguageModelConfig, MultimodalLanguageModel

#     >>> # Initializing a CLIPVisionModel and Llava configuration
#     >>> vision_config = CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch16")
#     >>> language_config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3-8B")

#     >>> # Initializing a MultimodalLanguageModel (with random weights)
#     >>> config = MultimodalLanguageModelConfig(text_config=language_config, vision_config=vision_config)
#     >>> model = MultimodalLanguageModel(config=config)

#     >>> # Initializing a MultimodalLanguageModel from pretrained vision and text models
#     >>> model = MultimodalLanguageModel.from_encoders_llm_pretrained("meta-llama/Meta-Llama-3-8B", "openai/clip-vit-base-patch16")

#     >>> # Accessing the model configuration
#     >>> config_vision = model.config.encoder_configs[0]
#     >>> config_text = model.config.text_config
#     ```
#     """

#     model_type = "multimodal-language-model"

#     # This config class is composed of multiple sub-configs
#     is_composition = True

#     def __init__(
#         self,
#         text_config: PretrainedConfig,
#         vision_config: PretrainedConfig,
#         vision_projector_config: MultimodalLanguageModelProjectorConfig,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         vision_model_type = vision_config.model_type
#         vision_config_class = VISION_MODEL_CONFIGS.get(vision_model_type, None)
#         if vision_config_class is None:
#             raise ValueError(f"Unsupported vision config: {vision_model_type}")

#         self.text_config = text_config
#         self.vision_config = vision_config
#         self.vision_projector_config = vision_projector_config
