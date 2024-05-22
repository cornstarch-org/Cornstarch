from transformers import AutoConfig, AutoModel

from cornstarch.models.evaclip import (
    EvaCLIPConfig,
    EvaCLIPPreTrainedModel,
    EvaCLIPVisionConfig,
    EvaCLIPVisionModel,
)
from cornstarch.models.intern_vit import InternVisionConfig, InternVisionModel

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

AutoModel.register(EvaCLIPConfig, EvaCLIPPreTrainedModel)
AutoModel.register(EvaCLIPVisionConfig, EvaCLIPVisionModel)

AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)
