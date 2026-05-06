from transformers import AutoConfig, AutoModel

from cornstarch.models.evaclip import (
    EvaCLIPConfig,
    EvaCLIPPreTrainedModel,
    EvaCLIPVisionConfig,
    EvaCLIPVisionModel,
)
from cornstarch.models.intern_vit import InternVisionConfig, InternVisionModel

AutoModel.register(EvaCLIPConfig, EvaCLIPPreTrainedModel)
AutoModel.register(EvaCLIPVisionConfig, EvaCLIPVisionModel)

AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)
