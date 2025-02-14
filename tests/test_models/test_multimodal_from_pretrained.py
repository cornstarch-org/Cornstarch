import tempfile
from pathlib import Path
from typing import Type

import pytest
import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    PreTrainedModel,
)
from transformers.models.llava_next import (
    LlavaNextForConditionalGeneration,
)
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)

url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image_stop = Image.open(requests.get(url, stream=True).raw)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_cats = Image.open(requests.get(url, stream=True).raw)

url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.jpg"
image_snowman = Image.open(requests.get(url, stream=True).raw)

images = [image_stop, image_cats, image_snowman]


@pytest.fixture(scope="module")
def temp_directory():
    with tempfile.TemporaryDirectory():
        yield Path(tempfile.gettempdir())


@pytest.mark.parametrize(
    "model_name, model_cls",
    [("llava-hf/llava-v1.6-vicuna-7b-hf", LlavaNextForConditionalGeneration)],
    ids=["llava-v1.6-vicuna-7b-hf"],
)
@pytest.mark.parametrize("batch_size", [1, 2], ids=lambda x: f"bs{x}")
def test_llava_model_generation(
    model_name: str,
    model_cls: Type[PreTrainedModel],
    batch_size: int,
    temp_directory: Path,
):
    prompts = [
        "USER: <image>\nWhat is shown in this image? ASSISTANT:",
        "USER: <image>\nWhat about this image? How many cats do you see ASSISTANT:",
        "USER: <image>\nWhat is shown in this image? ASSISTANT:",
    ]

    cornstarch_model: MultimodalModel = (
        MultimodalModel.from_pretrained_multimodal_model(
            pretrained_model_id=model_name,
            cache_dir=temp_directory,
        ).to(dtype=torch.bfloat16, device="cuda")
    )
    cornstarch_model.train(encoders_mode={"vision": (False, False)}, llm_mode=False)

    hf_model: PreTrainedModel = model_cls.from_pretrained(
        model_name, cache_dir=temp_directory
    ).to(dtype=torch.bfloat16, device="cuda")
    hf_model.train(mode=False)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=temp_directory)
    processor.tokenizer.padding_side = "left"
    processor.patch_size = hf_model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = (
        hf_model.config.vision_feature_select_strategy
    )

    # llava text generation
    inputs = processor(
        images[:batch_size], prompts[:batch_size], padding=True, return_tensors="pt"
    ).to(dtype=torch.bfloat16, device="cuda")

    hf_output = hf_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    hf_output_text = processor.batch_decode(hf_output, skip_special_tokens=True)

    # cornstarch text generation
    cornstarch_output = cornstarch_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    cornstarch_output_text = processor.batch_decode(
        cornstarch_output, skip_special_tokens=True
    )

    assert hf_output_text == cornstarch_output_text


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2-VL-2B-Instruct"])
@pytest.mark.parametrize("batch_size", [1, 2], ids=lambda x: f"bs{x}")
def test_qwen2vl_model_generation(
    model_name: str,
    batch_size: int,
    temp_directory: Path,
):
    cornstarch_model: MultimodalModel = (
        MultimodalModel.from_pretrained_multimodal_model(
            pretrained_model_id=model_name,
            cache_dir=temp_directory,
            trust_remote_code=False,
        ).to(dtype=torch.bfloat16, device="cuda")
    )
    cornstarch_model.train(encoders_mode={"vision": (False, False)}, llm_mode=False)

    hf_model: Qwen2VLForConditionalGeneration = (
        Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, cache_dir=temp_directory, trust_remote_code=False
        ).to(dtype=torch.bfloat16, device="cuda")
    )
    hf_model.train(mode=False)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=temp_directory)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        conversation, tokenizer=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt] * batch_size,
        images=images[:batch_size],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    # qwen2vl text generation
    hf_output = hf_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    hf_output = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, hf_output)
    ]
    hf_output_text = processor.batch_decode(
        hf_output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # cornstarch text generation
    cornstarch_output = cornstarch_model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    cornstarch_output = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, cornstarch_output)
    ]
    cornstarch_output_text = processor.batch_decode(
        cornstarch_output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    assert hf_output_text == cornstarch_output_text
