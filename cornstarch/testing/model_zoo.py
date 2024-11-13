import copy
from abc import ABC, abstractmethod
from typing import Type

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Config,
    Dinov2Model,
)
from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralForCausalLM,
)
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionConfig,
    PixtralVisionModel,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioEncoder,
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLVisionConfig,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.intern_vit.modeling_intern_vit import (
    InternVisionConfig,
    InternVisionModel,
)
from cornstarch.models.internlm2.modeling_internlm2 import (
    InternLM2Config,
    InternLM2ForCausalLM,
)

from .utils import create_random_image


class ModelClassBase(ABC):
    def __init__(self, model_class: Type[PreTrainedModel], config: PretrainedConfig):
        self.model_class = model_class
        self.config = config

    def build_model(self) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.tie_word_embeddings = False
        model = self.model_class(config)
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            pass

        return model

    @abstractmethod
    def data(self, batch_size: int, **kwargs) -> dict[str, torch.Tensor]: ...


class LanguageModelClassBase(ModelClassBase):
    def data(self, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
        device = torch.device("cuda")
        data = {
            "input_ids": torch.randint(
                200,
                self.config.vocab_size,
                (batch_size, seq_len),
                dtype=torch.long,
                device=device,
                requires_grad=False,
            ),
            # "attention_mask": torch.ones((batch_size, seq_len), device=device),
        }
        data["labels"] = data["input_ids"].clone()

        return data


class ImageModelClassBase(ModelClassBase):
    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        device = torch.device("cuda")
        data = {
            "pixel_values": torch.randn(
                batch_size,
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=False,
            )
        }

        return data


class AudioModelClassBase(ModelClassBase):
    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        device = torch.device("cuda")
        data = {
            "input_features": torch.randn(
                batch_size,
                self.config.num_mel_bins,
                self.config.max_source_positions * 2,
                dtype=torch.bfloat16,
                device=device,
                requires_grad=False,
            )
        }

        return data


class Gemma2bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            GemmaForCausalLM,
            GemmaConfig.from_pretrained("google/gemma-2b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Gemma7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            GemmaForCausalLM,
            GemmaConfig.from_pretrained("google/gemma-1.1-7b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Llama8bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
        )


class Vicuna7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("lmsys/vicuna-7b-v1.5"),
        )
        self.config._attn_implementation = "flash_attention_2"


class InternLM27bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            InternLM2ForCausalLM,
            InternLM2Config.from_pretrained("internlm/internlm2_5-7b-chat-1m"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Mistral7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            MistralForCausalLM,
            MistralConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Phi3MiniClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Phi3ForCausalLM,
            Phi3Config.from_pretrained("microsoft/Phi-3.5-mini-instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Qwen27bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
        )


class CLIPVisionClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            CLIPVisionModel,
            CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336"),
        )
        self.config._attn_implementation = "flash_attention_2"


class SiglipVisionClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            SiglipVisionModel,
            SiglipVisionConfig.from_pretrained("google/siglip-so400m-patch14-384"),
        )
        self.config._attn_implementation = "flash_attention_2"


class InternVision300mClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            InternVisionModel,
            InternVisionConfig.from_pretrained("OpenGVLab/InternViT-300M-448px"),
        )
        self.config._attn_implementation = "eager"
        self.config.qk_normalization = False


class InternVision6bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            InternVisionModel,
            InternVisionConfig.from_pretrained("OpenGVLab/InternViT-6B-448px-V1-5"),
        )
        self.config._attn_implementation = "eager"
        self.config.qk_normalization = False
        self.config.num_attention_heads = 24  # originally 25
        self.config.hidden_size = 3072  # originally 3200. Need to be multiple of 24


class PixtralVisionClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            PixtralVisionModel,
            PixtralVisionConfig.from_pretrained("mistral-community/pixtral-12b"),
        )
        self.config._attn_implementation = "eager"

    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        from transformers.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        image = create_random_image(1280, 720)
        processor = PixtralImageProcessor.from_pretrained(
            "mistral-community/pixtral-12b"
        )
        return processor(images=[image] * batch_size, return_tensors="pt").to("cuda")


class Dinov2LargeClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config.from_pretrained("facebook/dinov2-large"),
        )
        self.config._attn_implementation = "eager"


class Qwen2Vision7bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2VisionTransformerPretrainedModel,
            Qwen2VLVisionConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"

    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        image = create_random_image(720, 480)
        processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return processor(images=[image] * batch_size, return_tensors="pt").to("cuda")


class Qwen2AudioEncoderClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2AudioEncoder,
            Qwen2AudioEncoderConfig.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class WhisperClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-large-v3"),
        )
        self.config._attn_implementation = "flash_attention_2"
