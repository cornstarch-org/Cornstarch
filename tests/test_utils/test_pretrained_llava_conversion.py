import pathlib
from typing import Type

import pytest
import torch
import requests
from PIL import Image
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

from cornstarch.models.multimodal_language_model import (
    ModalModule,
    MultimodalModel,
    MultimodalProjector,
)

vision_model_name_or_path_list = [
    ("openai/clip-vit-large-patch14-336", CLIPVisionConfig, CLIPVisionModel),
]

language_model_name_or_path_list = [
    "lmsys/vicuna-7b-v1.5",
]

pretrained_model_name_or_path_list = ["llava-hf/llava-1.5-7b-hf"]


@pytest.mark.parametrize(
    "vision_model_name_or_path", vision_model_name_or_path_list, ids=lambda x: x[0]
)
@pytest.mark.parametrize("text_model_name_or_path", language_model_name_or_path_list)
@pytest.mark.parametrize(
    "pretrained_model_name_or_path", pretrained_model_name_or_path_list
)
def test_multimodal_model_generation(
    vision_model_name_or_path: tuple[
        str, Type[PretrainedConfig], Type[PreTrainedModel]
    ],
    text_model_name_or_path: str,
    pretrained_model_name_or_path: str,
    tmp_path: pathlib.Path,
):
    # create cornstarch llava model
    cornstarch_llava = MultimodalModel.from_pretrained_multimodal_model(
        model_id=pretrained_model_name_or_path,
    ).to(torch.float16)

    llava_model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        torch_dtype="auto",
        device_map="cuda",
    )

    # ToDo: compare weights

    pass
