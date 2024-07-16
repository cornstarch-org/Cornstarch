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

pretrained_model_name_or_path_list = [
    "llava-hf/llava-1.5-7b-hf"
]

@pytest.mark.parametrize(
    "vision_model_name_or_path", vision_model_name_or_path_list, ids=lambda x: x[0]
)
@pytest.mark.parametrize("text_model_name_or_path", language_model_name_or_path_list)
@pytest.mark.parametrize("pretrained_model_name_or_path", pretrained_model_name_or_path_list)
def test_multimodal_model_generation(
    vision_model_name_or_path: tuple[
        str, Type[PretrainedConfig], Type[PreTrainedModel]
    ],
    text_model_name_or_path: str,
    pretrained_model_name_or_path: str,
    tmp_path: pathlib.Path,
):
   
    # For faster testing, we do not initialize weights for the vision and language models
    vision_config = vision_model_name_or_path[1].from_pretrained(
        vision_model_name_or_path[0]
    )
    language_config = AutoConfig.from_pretrained(
        text_model_name_or_path, trust_remote_code=True
    )

    vision_model = vision_model_name_or_path[2](vision_config)

    with init_empty_weights():
        # Parameters of language models are not necessary in this test
        # Use meta device to initialize the model.
        language_model = AutoModelForCausalLM.from_config(
            language_config, trust_remote_code=True
        )

    cornstarch_llava = MultimodalModel.from_pretrained_multimodal_model(
        model_id=pretrained_model_name_or_path,
        encoders={"vision": ModalModule(vision_model)},
        language_model=language_model,
    ).to(torch.float16)

    llava_model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    )

    # ToDo: compare weights