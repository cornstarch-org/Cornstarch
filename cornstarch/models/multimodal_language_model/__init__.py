from transformers import AutoConfig, AutoModel

from .configuration_multimodal_language_model import MultimodalProjectorConfig
from .modeling_multimodal_language_model import (
    ModalDecoderModule,
    ModalEncoderModule,
    ModalModuleBase,
    MultimodalModel,
    MultimodalProjector,
)
from .processing_multimodal_language_model import MultimodalProcessor

AutoConfig.register("multimodal-projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)
