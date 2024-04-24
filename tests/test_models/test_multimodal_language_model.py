from typing import Type

import pytest
import transformers
from accelerate import init_empty_weights
from packaging.version import Version
from transformers import AutoConfig, PretrainedConfig
from transformers.models.clip import CLIPVisionConfig
from transformers.models.dinov2 import Dinov2Config

from cornstarch.models.multimodal_language_model import (
    MultimodalLanguageModel,
    MultimodalLanguageModelConfig,
)

vision_model_names = [
    ["openai/clip-vit-base-patch16", CLIPVisionConfig],
    ["laion/CLIP-ViT-B-32-laion2B-s34B-b79k", CLIPVisionConfig],
    ["BAAI/EVA-CLIP-18B", CLIPVisionConfig],
    ["facebook/dinov2-giant", Dinov2Config],
]

language_model_names = [
    "google/gemma-7b",
    "meta-llama/Meta-Llama-3-8B",
    "facebook/opt-66b",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3-mini-4k-instruct",
]


@pytest.mark.parametrize("vision_model_name", vision_model_names, ids=lambda x: x[0])
@pytest.mark.parametrize("language_model_name", language_model_names)
def test_build_empty_model(
    vision_model_name: tuple[str, Type[PretrainedConfig]], language_model_name: str
):
    if "gemma" in language_model_name and Version(transformers.__version__) < Version(
        "4.38.0"
    ):
        pytest.skip("Google gemma models are not supported in transformers < 4.38.0")

    vision_config = vision_model_name[1].from_pretrained(vision_model_name[0])
    language_config = AutoConfig.from_pretrained(
        language_model_name, trust_remote_code=True
    )

    mm_config = MultimodalLanguageModelConfig(
        text_config=language_config, vision_config=vision_config
    )
    with init_empty_weights():
        model = MultimodalLanguageModel(config=mm_config, trust_remote_code=True)
