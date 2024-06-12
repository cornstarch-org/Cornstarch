from transformers import AutoConfig, AutoModel

from .configuration_multimodal_language_model import MultimodalProjectorConfig
from .modeling_multimodal_language_model import (
    ModalModule,
    MultimodalModel,
    MultimodalProjector,
)
from .processing_multimodal_language_model import MultimodalLanguageModelProcessor

AutoConfig.register("multimodal-projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)
