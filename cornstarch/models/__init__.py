from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from cornstarch.models.evaclip import (
    EvaCLIPConfig,
    EvaCLIPPreTrainedModel,
    EvaCLIPVisionConfig,
    EvaCLIPVisionModel,
)
from cornstarch.models.intern_vit import InternVisionConfig, InternVisionModel
from cornstarch.models.internlm import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2Model,
    InternLM2Tokenizer,
    InternLM2TokenizerFast,
)

AutoModel.register(EvaCLIPConfig, EvaCLIPPreTrainedModel)
AutoModel.register(EvaCLIPVisionConfig, EvaCLIPVisionModel)

AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)

AutoConfig.register("internlm2", InternLM2Config)
AutoModel.register(InternLM2Config, InternLM2Model)
AutoModelForCausalLM.register(InternLM2Config, InternLM2ForCausalLM)
AutoTokenizer.register(InternLM2Config, InternLM2Tokenizer, InternLM2TokenizerFast)
