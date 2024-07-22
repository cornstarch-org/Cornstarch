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
    ).to(dtype=torch.float16, device="cuda")

    llava_model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        torch_dtype="auto",
        device_map="cuda",
    )
    llava_model.config.vision_feature_layer = -1
    llava_model.config.vision_feature_select_strategy = "full"

    # ToDo: compare weights
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"

    llava_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    prompt = "<image>USER: What are these? ASSISTANT:"
    llava_inputs = llava_processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )
    llava_inputs["input_ids"][0][0], llava_inputs["input_ids"][0][1] = llava_inputs["input_ids"][0][1].clone(), llava_inputs["input_ids"][0][0].clone()
    
    llava_output = llava_model.generate(
        **llava_inputs,
        max_new_tokens=20,
        do_sample=False,
        vision_feature_layer=-1,
        pad_token_id=llava_processor.tokenizer.eos_token_id,
    )
    print(llava_processor.decode(llava_output[0][2:], skip_special_tokens=True))

    prompt = "USER: What are these? ASSISTANT:"
    cornstarch_inputs = llava_processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )
    cornstarch_output = cornstarch_llava.generate(
        **cornstarch_inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=llava_processor.tokenizer.eos_token_id,
    )
    print(llava_processor.decode(cornstarch_output[0], skip_special_tokens=True))

    # print("***", llava_model.config.vision_feature_layer, llava_model.config.vision_feature_select_strategy)
    # print("!!!", cornstarch_output)

    
