from transformers.models.clip import CLIPVisionConfig
from transformers.models.dinov2 import Dinov2Config
from transformers.models.gemma import GemmaConfig
from transformers.models.gemma2 import Gemma2Config
from transformers.models.llama import LlamaConfig
from transformers.models.mistral import MistralConfig
from transformers.models.mixtral import MixtralConfig
from transformers.models.phi3 import Phi3Config
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2_audio import Qwen2AudioConfig
from transformers.models.siglip import SiglipVisionConfig
from transformers.models.whisper import WhisperConfig

from cornstarch.models.evaclip import EvaCLIPVisionConfig
from cornstarch.models.intern_vit import InternVisionConfig
from cornstarch.models.internlm2 import InternLM2Config

llama_config = LlamaConfig()
gemma_config = GemmaConfig()
gemma2_config = Gemma2Config()
internlm2_config = InternLM2Config()
mistral_config = MistralConfig()
mixtral_config = MixtralConfig()
phi3_config = Phi3Config()
qwen2_config = Qwen2Config()

for language_config in [
    llama_config,
    gemma_config,
    gemma2_config,
    mistral_config,
    mixtral_config,
    phi3_config,
    qwen2_config,
    internlm2_config,
]:
    language_config.hidden_size = 256
    language_config.intermediate_size = 256
    language_config.num_attention_heads = 16
    language_config.num_key_value_heads = 16
    language_config.num_hidden_layers = 4
    language_config.use_cache = False
    language_config._attn_implementation = "eager"
    # TODO: Gemma uses tie_word_embeddings True, in which case the tests fail.
    # Implement automatic gradient synchronization between tied weights.
    # Existing explicit synchronization is not enough as there are encoders
    # that need to have gradients propagated "after" the weights are synchronized.
    language_config.tie_word_embeddings = False

# GQA adjustment. Models not in this list use MHA.
for language_config in [
    gemma2_config,
    internlm2_config,
    mistral_config,
    mixtral_config,
    qwen2_config,
]:
    language_config.num_key_value_heads = 8

# MoE adjustment
for language_config in [mixtral_config]:
    language_config.num_local_experts = 4
    language_config.num_experts_per_tok = 1


clip_config = CLIPVisionConfig()
siglip_config = SiglipVisionConfig()
dinov2_config = Dinov2Config()
evaclip_config = EvaCLIPVisionConfig()
internvit_config = InternVisionConfig()

for vision_config in [
    clip_config,
    siglip_config,
    dinov2_config,
    evaclip_config,
    internvit_config,
]:
    vision_config.hidden_size = 256
    vision_config.intermediate_size = 256
    vision_config.num_attention_heads = 8
    vision_config.num_hidden_layers = 4
    vision_config._attn_implementation = "eager"
    vision_config.use_cache = False

whisper_config = WhisperConfig()
qwen2audio_config = Qwen2AudioConfig()
