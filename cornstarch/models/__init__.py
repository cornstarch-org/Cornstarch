from transformers import AutoModel

from cornstarch.models.evaclip import (
    EvaCLIPConfig,
    EvaCLIPPreTrainedModel,
    EvaCLIPVisionConfig,
    EvaCLIPVisionModel,
)

AutoModel.register(EvaCLIPConfig, EvaCLIPPreTrainedModel)
AutoModel.register(EvaCLIPVisionConfig, EvaCLIPVisionModel)
