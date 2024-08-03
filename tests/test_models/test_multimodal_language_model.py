from typing import Type

import pytest
import transformers
from accelerate import init_empty_weights
from packaging.version import Version
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers.models.dinov2 import Dinov2Config, Dinov2Model
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)

vision_model = [
    ("openai/clip-vit-base-patch16", CLIPVisionConfig, CLIPVisionModel),
    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79k", CLIPVisionConfig, CLIPVisionModel),
    ("BAAI/EVA-CLIP-18B", CLIPVisionConfig, CLIPVisionModel),
    ("facebook/dinov2-giant", Dinov2Config, Dinov2Model),
]

audio_model = [
    ("openai/whisper-small", WhisperConfig, WhisperEncoder),
]

language_model = [
    "google/gemma-7b",
    "meta-llama/Meta-Llama-3-8B",
    "facebook/opt-66b",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct",
]


@pytest.mark.parametrize("audio_model", audio_model, ids=lambda x: x[0])
@pytest.mark.parametrize("language_model", language_model)
def test_build_alm(
    audio_model: tuple[str, Type[PretrainedConfig], Type[PreTrainedModel]],
    language_model: str,
):
    with init_empty_weights():
        audio_config = audio_model[1].from_pretrained(audio_model[0])
        audio_model = audio_model[2](audio_config)

        language_config = AutoConfig.from_pretrained(
            language_model, trust_remote_code=False
        )
        language_model = AutoModelForCausalLM.from_config(
            language_config, trust_remote_code=False
        )

        MultimodalModel(
            encoders={"audio": ModalModule(audio_model)},
            language_model=language_model,
        )


@pytest.mark.parametrize("vision_model", vision_model, ids=lambda x: x[0])
@pytest.mark.parametrize("language_model", language_model)
def test_build_vlm(
    vision_model: tuple[str, Type[PretrainedConfig], Type[PreTrainedModel]],
    language_model: str,
):
    if "gemma" in language_model and Version(transformers.__version__) < Version(
        "4.38.0"
    ):
        pytest.skip("Google gemma models are not supported in transformers < 4.38.0")

    with init_empty_weights():
        vision_config = vision_model[1].from_pretrained(vision_model[0])
        vision_model = vision_model[2](vision_config)

        language_config = AutoConfig.from_pretrained(
            language_model, trust_remote_code=False
        )
        language_model = AutoModelForCausalLM.from_config(
            language_config, trust_remote_code=False
        )

        MultimodalModel(
            encoders={"vision": ModalEncoderModule(vision_model)},
            language_model=language_model,
        )
