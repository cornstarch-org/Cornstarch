from transformers import AutoModel

from .configuration_multimodal_language_model import (
    MultimodalLanguageModelConfig,
    MultimodalLanguageModelProjectorConfig,
)
from .modeling_multimodal_language_model import (
    MultimodalLanguageModel,
    MultimodalProjectorModel,
)
from .processing_multimodal_language_model import MultimodalLanguageModelProcessor

AutoModel.register(MultimodalLanguageModelProjectorConfig, MultimodalProjectorModel)
