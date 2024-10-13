from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from cornstarch.models.evaclip import (
    EvaCLIPConfig,
    EvaCLIPPreTrainedModel,
    EvaCLIPVisionConfig,
    EvaCLIPVisionModel,
)
from cornstarch.models.intern_vit import InternVisionConfig, InternVisionModel
from cornstarch.models.internlm2 import (
    InternLM2Config,
    InternLM2ForCausalLM,
    InternLM2Model,
    InternLM2Tokenizer,
    InternLM2TokenizerFast,
)
from cornstarch.models.internvl2 import InternVLChatConfig, InternVLChatModel

AutoModel.register(EvaCLIPConfig, EvaCLIPPreTrainedModel)
AutoModel.register(EvaCLIPVisionConfig, EvaCLIPVisionModel)

AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)

AutoConfig.register("internlm2", InternLM2Config)
AutoModel.register(InternLM2Config, InternLM2Model)
AutoModelForCausalLM.register(InternLM2Config, InternLM2ForCausalLM)
AutoTokenizer.register(InternLM2Config, InternLM2Tokenizer, InternLM2TokenizerFast)

AutoConfig.register("internvl_chat", InternVLChatConfig)
AutoModel.register(InternVLChatConfig, InternVLChatModel)
AutoTokenizer.register(InternVLChatConfig, InternLM2Tokenizer, InternLM2TokenizerFast)
