from __future__ import annotations

import copy
import functools
import math
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
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralConfig,
    MistralForCausalLM,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralForCausalLM,
)
from transformers.models.phi3.modeling_phi3 import Phi3Config, Phi3ForCausalLM
from transformers.models.pixtral.modeling_pixtral import (
    PixtralVisionConfig,
    PixtralVisionModel,
)
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioConfig,
    Qwen2AudioEncoder,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLVisionConfig,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.vit.modeling_vit import ViTConfig, ViTModel
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
        config.use_cache = False
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
    def data(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        data = {
            "pixel_values": torch.randn(
                self.get_num_tokens(batch_size, image_size),
                self.config.num_channels,
                self.config.image_size,
                self.config.image_size,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
                requires_grad=False,
            )
        }

        return data

    def get_num_tokens(self, batch_size: int, image_size: tuple[int, int]) -> int:
        num_chunks = math.ceil(image_size[0] / self.config.image_size) * math.ceil(
            image_size[1] / self.config.image_size
        )
        return batch_size * num_chunks

    def build_model(self) -> ModalEncoderModule:
        return ModalEncoderModule(super().build_model())


class AudioModelClassBase(ModelClassBase):
    def data(self, batch_size: int) -> dict[str, torch.Tensor]:
        data = {
            "input_features": torch.randn(
                batch_size,
                self.config.num_mel_bins,
                self.config.max_source_positions * 2,
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
                requires_grad=False,
            )
        }

        return data

    def get_num_tokens(self) -> int:
        return self.config.max_source_positions

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
        self.config.cache_implementation = None


class Gemma7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            GemmaForCausalLM,
            GemmaConfig.from_pretrained("google/gemma-1.1-7b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.cache_implementation = None


class Gemma227bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Gemma2ForCausalLM,
            Gemma2Config.from_pretrained("google/gemma-2-27b-it"),
        )
        self.config._attn_implementation = "flash_attention_2"
        self.config.cache_implementation = None


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

    def data(self, batch_size: int, seq_len: int) -> dict[str, torch.Tensor]:
        data = super().data(batch_size, seq_len)
        data["use_cache"] = torch.tensor([False] * batch_size, dtype=torch.bool)

        return data


class Mistral123bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            MixtralForCausalLM,
            MixtralConfig.from_pretrained("SillyTilly/Mistral-Large-Instruct-2407"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Mixtral8x7bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            MixtralForCausalLM,
            MixtralConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Mixtral8x22bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            MixtralForCausalLM,
            MixtralConfig.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1"),
        )
        self.config._attn_implementation = "flash_attention_2"


class Phi3MiniClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Phi3ForCausalLM,
            Phi3Config.from_pretrained("microsoft/Phi-3.5-mini-instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"
        # FlexAttention doesn't support non-power-of-2 head_dim as of now.
        # Fix hidden_size to make it work
        self.config.hidden_size = 4096  # 32 * 128


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


class Qwen232bClass(LanguageModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2ForCausalLM,
            Qwen2Config.from_pretrained("Qwen/Qwen2.5-32B-Instruct"),
        )
        self.config._attn_implementation = "flash_attention_2"


class ViT22bClass(ImageModelClassBase):
    def __init__(self):
        super().__init__(
            ViTModel,
            ViTConfig(
                hidden_size=6144,
                num_hidden_layers=48,
                intermediate_size=24576,
                num_attention_heads=48,
                image_size=448,
                patch_size=14,
            ),
        )
        self.config._attn_implementation = "eager"


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
        self.config.image_size = 448


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
        self.data_shape = None

    def data(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        from transformers.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        image = create_random_image(*image_size)
        processor = PixtralImageProcessor.from_pretrained(
            "mistral-community/pixtral-12b"
        )

        inputs = processor(images=image, return_tensors="pt").to("cuda")
        self.data_shape = inputs["pixel_values"][0].shape
        """
        Pixtral doesn't seem to support batch processing, hence
        here we do some data preprocessing to make it work.
        """
        inputs["pixel_values"] = torch.cat(
            [pixel_value.unsqueeze(0) for pixel_value in inputs["pixel_values"]], dim=0
        ).repeat(batch_size, 1, 1, 1)

        return inputs

    def get_num_tokens(self, batch_size: int, image_size: tuple[int, int]) -> int:
        return (self.data_shape[-1] // self.config.patch_size) * (
            self.data_shape[-2] // self.config.patch_size
        )


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
        self.config.image_size = 518
        self.config.patch_size = 14


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
        self.data_shape = None

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
            postprocess_module_callback=Qwen2VLModel.postprocess_vision_callback,
        )

    def data(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        image = create_random_image(*image_size)
        processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        inputs = processor(images=image, return_tensors="pt").to("cuda")
        self.data_shape = inputs["pixel_values"].shape
        """
        Preprocess inputs to make it work with micro batching.
        """
        for key, value in inputs.items():
            inputs[key] = torch.cat([value.unsqueeze(0)] * batch_size, dim=0)

        return inputs

    def get_num_tokens(self, batch_size: int, image_size: tuple[int, int]) -> int:
        return self.data_shape[0] // (self.config.temporal_patch_size**2)


class Qwen2AudioEncoderClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            Qwen2AudioEncoder,
            Qwen2AudioConfig.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct"
            ).audio_config,
        )
        self.config._attn_implementation = "eager"

    def get_num_tokens(self) -> int:
        return self.config.max_source_positions // 2


class WhisperLargeClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-large-v3"),
        )
        self.config._attn_implementation = "eager"


class WhisperMediumClass(AudioModelClassBase):
    def __init__(self):
        super().__init__(
            WhisperEncoder,
            WhisperConfig.from_pretrained("openai/whisper-medium"),
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
    "gemma2_27b": Gemma227bClass,
    "llama_1b": Llama1bClass,
    "llama_3b": Llama3bClass,
    "llama_8b": Llama8bClass,
    "llama_70b": Llama70bClass,
    "internlm2_1.8b": InternLM218bClass,
    "internlm2_8b": InternLM28bClass,
    "internlm2_20b": InternLM220bClass,
    "mistral_7b": Mistral7bClass,
    "mistral_123b": Mistral123bClass,
    "mixtral_8x7b": Mixtral8x7bClass,
    "mixtral_8x22b": Mixtral8x22bClass,
    "phi3_mini": Phi3MiniClass,
    "phi3_small": Phi3SmallClass,
    "qwen2_0.5b": Qwen205bClass,
    "qwen2_1.5b": Qwen215bClass,
    "qwen2_3b": Qwen23bClass,
    "qwen2_7b": Qwen27bClass,
    "qwen2_14b": Qwen214bClass,
    "qwen2_32b": Qwen232bClass,
    "qwen2_72b": Qwen272bClass,
    "vicuna": Vicuna7bClass,
    "vit_22b": ViT22bClass,
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
    "whisper_307m": WhisperMediumClass,
    "whisper_72m": WhisperBaseClass,
    "qwen2_audio": Qwen2AudioEncoderClass,
}

class_to_forward_str = {
    Gemma2bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Gemma7bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Gemma227bClass: "cornstarch.shardformer.modeling.gemma2.Gemma2ModelForwards.gemma2_model_forward",
    Llama1bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama3bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama8bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama70bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    InternLM218bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    InternLM28bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    InternLM220bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    Mistral7bClass: "cornstarch.shardformer.modeling.mistral.MistralModelForwards.mistral_model_forward",
    Mistral123bClass: "cornstarch.shardformer.modeling.mistral.MistralModelForwards.mistral_model_forward",
    Mixtral8x7bClass: "cornstarch.shardformer.modeling.mixtral.MixtralModelForwards.mixtral_model_forward",
    Mixtral8x22bClass: "cornstarch.shardformer.modeling.mixtral.MixtralModelForwards.mixtral_model_forward",
    Phi3MiniClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Phi3SmallClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Qwen205bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen215bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen23bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen27bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen214bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen232bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen272bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Vicuna7bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    ViT22bClass: "cornstarch.shardformer.modeling.vit.ViTModelForwards.vit_model_forward",
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
    WhisperMediumClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    WhisperLargeClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    Qwen2AudioEncoderClass: "cornstarch.shardformer.modeling.qwen2_audio.Qwen2AudioModelForwards.qwen2_audio_encoder_forward",
}
