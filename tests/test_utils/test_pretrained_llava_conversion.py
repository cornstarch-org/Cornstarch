import pathlib
from typing import Type

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)


pretrained_model_name_or_path_list = [
    "llava-hf/llava-1.5-7b-hf"
]

@pytest.mark.parametrize(
    "pretrained_model_name_or_path", pretrained_model_name_or_path_list
)
def test_multimodal_model_generation(
    pretrained_model_name_or_path: str,
):
    # create cornstarch llava model
    cornstarch_llava = MultimodalModel.from_pretrained_multimodal_model(
        pretrained_model_id=pretrained_model_name_or_path,
    ).to(dtype=torch.float16, device="cuda")

    # create llava model
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        revision="main",
        torch_dtype="auto",
        device_map="cuda",
    )

    # config vision feature layer to match cornstarch settings
    llava_model.config.vision_feature_layer = -1
    llava_model.config.vision_feature_select_strategy = "full"

    llava_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    # loading sample image file
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # llava text generation
    prompt = "<image>USER: What are these? ASSISTANT:"
    llava_inputs = llava_processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )
    llava_inputs["input_ids"][0][0], llava_inputs["input_ids"][0][1] = (
        llava_inputs["input_ids"][0][1].clone(),
        llava_inputs["input_ids"][0][0].clone(),
    )

    llava_output = llava_model.generate(
        **llava_inputs,
        max_new_tokens=20,
        do_sample=False,
        vision_feature_layer=-1,
        pad_token_id=llava_processor.tokenizer.eos_token_id,
    )
    llava_text_output = (
        llava_processor.decode(llava_output[0][2:], skip_special_tokens=True)
        .split("ASSISTANT:")[-1]
        .strip()
    )

    # cornstarch text generation
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
    cornstarch_text_output = llava_processor.decode(
        cornstarch_output[0], skip_special_tokens=True
    )

    assert llava_text_output == cornstarch_text_output
