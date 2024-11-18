from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from types import MethodType
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

from cornstarch.models.evaclip.modeling_evaclip import (
    EvaCLIPVisionModel,
)
from cornstarch.models.intern_vit.modeling_intern_vit import (
    InternVisionConfig,
    InternVisionModel,
)
from cornstarch.models.internlm2.modeling_internlm2 import (
    InternLM2Config,
    InternLM2ForCausalLM,
)
from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
)
from cornstarch.models.multimodal_language_model.modeling_multimodal_language_model import (
    Qwen2VLModel,
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

    def build_model(self) -> ModalEncoderModule:
        return ModalEncoderModule(super().build_model())


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

    def build_model(self) -> ModalEncoderModule:
        return ModalEncoderModule(super().build_model())


class Gemma2bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            GemmaForCausalLM,
            GemmaConfig.from_pretrained("google/gemma-2b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.num_key_value_heads = 4


class Gemma7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            GemmaForCausalLM,
            GemmaConfig.from_pretrained("google/gemma-1.1-7b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Llama1bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.tie_word_embeddings = False


class Llama3bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("meta-llama/Llama-3.2-3B"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.tie_word_embeddings = False


class Llama8bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Llama70bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("meta-llama/Llama-3.1-70B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.num_hidden_layers = 40


class Vicuna7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            LlamaForCausalLM,
            LlamaConfig.from_pretrained("lmsys/vicuna-7b-v1.5"),
        )
        self.config._attn_implementation = "flash_attention_2"


class InternLM218bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            InternLM2ForCausalLM,
            InternLM2Config.from_pretrained("internlm/internlm2_5-1_8b"),
        )
        self.config._attn_implementation = "flash_attention_2"


class InternLM28bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            InternLM2ForCausalLM,
            InternLM2Config.from_pretrained("internlm/internlm2_5-1_8b"),
        )
        self.config._attn_implementation = "flash_attention_2"


class InternLM220bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            InternLM2ForCausalLM,
            InternLM2Config.from_pretrained("internlm/internlm2_5-20b"),
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


class Phi3SmallClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Phi3ForCausalLM,
            Phi3Config.from_pretrained("microsoft/Phi-3-small-8k-instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.hidden_act = "gelu"


class Qwen205bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-0.5B"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Qwen215bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Qwen23bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-3B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Qwen27bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Qwen272bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-72B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.num_hidden_layers = 40


class Qwen214bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-14B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class CLIPVisionClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            CLIPVisionModel,
            CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336"),
        )
        self.config._attn_implementation = "eager"


class EvaCLIPVision8bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            EvaCLIPVisionModel,
            CLIPVisionConfig.from_pretrained("BAAI/EVA-CLIP-8B-448"),
        )
        self.config._attn_implementation = "eager"


class EvaCLIPVision18bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            EvaCLIPVisionModel,
            CLIPVisionConfig.from_pretrained("BAAI/EVA-CLIP-18B"),
        )
        self.config._attn_implementation = "eager"


class SiglipVisionClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            SiglipVisionModel,
            SiglipVisionConfig.from_pretrained("google/siglip-so400m-patch14-384"),
        )
        self.config._attn_implementation = "eager"


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

        inputs = processor(images=image, return_tensors="pt").to("cuda")
        """
        Pixtral doesn't seem to support batch processing, hence
        here we do some data preprocessing to make it work.
        """
        inputs["pixel_values"] = torch.cat(
            [pixel_value.unsqueeze(0) for pixel_value in inputs["pixel_values"]], dim=0
        ).repeat(batch_size, 1, 1, 1)

        return inputs


class Dinov2GiantClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config.from_pretrained("facebook/dinov2-giant"),
        )
        self.config._attn_implementation = "eager"


class Dinov2LargeClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config.from_pretrained("facebook/dinov2-large"),
        )
        self.config._attn_implementation = "eager"


class Dinov2BaseClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config.from_pretrained("facebook/dinov2-base"),
        )
        self.config._attn_implementation = "eager"


class Dinov2SmallClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Dinov2Model,
            Dinov2Config.from_pretrained("facebook/dinov2-small"),
        )
        self.config._attn_implementation = "eager"


class Qwen2Vision7bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2VisionTransformerPretrainedModel,
            Qwen2VLVisionConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"),
        )
        self.config._attn_implementation = "eager"
        self.config.image_token_id = 44

    def build_model(self) -> PreTrainedModel:
        model: Qwen2VisionTransformerPretrainedModel = ModelClassBase.build_model(self)
        # projector = model.merger
        # model.merger = Qwen2VLModel.FakeMerger()
        model.forward = MethodType(
            functools.partial(
                Qwen2VLModel.vision_transformer_forward,
                original_forward=model.forward,
            ),
            model,
        )

        return ModalEncoderModule(
            model=model,
            additional_args=[
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            ],
            preprocess_callback=functools.partial(
                Qwen2VLModel.preprocess_vision_callback,
                visual_dtype=model.get_dtype(),
            ),
        )

    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        image = create_random_image(1280, 720)
        processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        return processor(images=[image] * batch_size, return_tensors="pt").to("cuda")


class Qwen2AudioEncoderClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2AudioEncoder,
            Qwen2AudioEncoderConfig.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct"),
        )
        self.config._attn_implementation = "eager"


class WhisperLargeClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-large-v3"),
        )
        self.config._attn_implementation = "eager"


class WhisperBaseClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-base"),
        )
        self.config._attn_implementation = "eager"


class WhisperSmallClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-small"),
        )
        self.config._attn_implementation = "eager"


model_to_class = {
    "gemma_2b": Gemma2bClass,
    "gemma_7b": Gemma7bClass,
    "llama_1b": Llama1bClass,
    "llama_3b": Llama3bClass,
    "llama_8b": Llama8bClass,
    "llama_70b": Llama70bClass,
    "internlm2_1.8b": InternLM218bClass,
    "internlm2_8b": InternLM28bClass,
    "internlm2_20b": InternLM220bClass,
    "mistral_7b": Mistral7bClass,
    "phi3_mini": Phi3MiniClass,
    "phi3_small": Phi3SmallClass,
    "qwen2_0.5b": Qwen205bClass,
    "qwen2_1.5b": Qwen215bClass,
    "qwen2_3b": Qwen23bClass,
    "qwen2_7b": Qwen27bClass,
    "qwen2_14b": Qwen214bClass,
    "qwen2_72b": Qwen272bClass,
    "vicuna": Vicuna7bClass,
    "clip": CLIPVisionClass,
    "evaclip_8b": EvaCLIPVision8bClass,
    "evaclip_18b": EvaCLIPVision18bClass,
    "dinov2_22m": Dinov2SmallClass,
    "dinov2_86m": Dinov2BaseClass,
    "dinov2_300m": Dinov2LargeClass,
    "dinov2_1.1b": Dinov2GiantClass,
    "intern_vit_300m": InternVision300mClass,
    "intern_vit_6b": InternVision6bClass,
    "pixtral_400m": PixtralVisionClass,
    "qwen2_vision_675m": Qwen2Vision7bClass,
    "siglip_878m": SiglipVisionClass,
    "whisper_1.5b": WhisperLargeClass,
    "whisper_242m": WhisperSmallClass,
    "whisper_72m": WhisperBaseClass,
    "qwen2_audio": Qwen2AudioEncoderClass,
}

class_to_forward_str = {
    Gemma2bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Gemma7bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Llama1bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama3bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama8bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama70bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    InternLM218bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    InternLM28bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    InternLM220bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    Mistral7bClass: "cornstarch.shardformer.modeling.mistral.MistralModelForwards.mistral_model_forward",
    Phi3MiniClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Phi3SmallClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Qwen205bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen215bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen23bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen27bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen214bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen272bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Vicuna7bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    CLIPVisionClass: "cornstarch.shardformer.modeling.clip.CLIPVisionModelForwards.clip_vision_transformer_forward",
    EvaCLIPVision8bClass: "cornstarch.shardformer.modeling.evaclip.EvaCLIPModelForwards.eva_clip_vision_transformer_forward",
    EvaCLIPVision18bClass: "cornstarch.shardformer.modeling.evaclip.EvaCLIPModelForwards.eva_clip_vision_transformer_forward",
    Dinov2SmallClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2BaseClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2LargeClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2GiantClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    InternVision300mClass: "cornstarch.shardformer.modeling.intern_vit.InternVisionModelForwards.intern_vit_model_forward",
    InternVision6bClass: "cornstarch.shardformer.modeling.intern_vit.InternVisionModelForwards.intern_vit_model_forward",
    PixtralVisionClass: "cornstarch.shardformer.modeling.pixtral.PixtralVisionModelForwards.pixtral_vision_model_forward",
    Qwen2Vision7bClass: "cornstarch.shardformer.modeling.qwen2_vision.Qwen2VisionModelForwards.qwen2_vision_transformer_forward",
    SiglipVisionClass: "cornstarch.shardformer.modeling.siglip.SiglipVisionModelForwards.siglip_vision_transformer_forward",
    WhisperSmallClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    WhisperBaseClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    WhisperLargeClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    Qwen2AudioEncoderClass: "cornstarch.shardformer.modeling.qwen2_audio.Qwen2AudioModelForwards.qwen2_audio_encoder_forward",
}
