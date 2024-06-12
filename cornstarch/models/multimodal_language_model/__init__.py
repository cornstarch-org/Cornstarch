from transformers import AutoConfig, AutoModel

from .configuration_multimodal_language_model import ProjectorModelConfig
from .modeling_multimodal_language_model import (
    ModalModule,
    ModalType,
    MultimodalModel,
    ProjectorModel,
)
from .processing_multimodal_language_model import MultimodalLanguageModelProcessor

AutoConfig.register("projector-model", ProjectorModelConfig)
AutoModel.register(ProjectorModelConfig, ProjectorModel)
