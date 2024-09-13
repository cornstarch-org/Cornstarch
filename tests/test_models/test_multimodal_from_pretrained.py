import copy
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
from transformers.models.llava import LlavaForConditionalGeneration
from transformers.models.llava_next import (
    LlavaNextForConditionalGeneration,
)

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
    [
        ("llava-hf/llava-1.5-7b-hf", LlavaForConditionalGeneration),
        ("llava-hf/llava-v1.6-vicuna-7b-hf", LlavaNextForConditionalGeneration),
    ],
    ids=["llava-1.5-7b-hf", "llava-v1.6-vicuna-7b-hf"],
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
        ).to(dtype=torch.float16, device="cuda")
    )
    cornstarch_model.train(mode=False)

    hf_model: PreTrainedModel = model_cls.from_pretrained(
        model_name, cache_dir=temp_directory
    ).to(dtype=torch.float16, device="cuda")
    hf_model.train(mode=False)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=temp_directory)
    processor.tokenizer.padding_side = "left"

    # llava text generation
    hf_inputs = processor(
        prompts[:batch_size], images[:batch_size], padding=True, return_tensors="pt"
    ).to(dtype=torch.float16, device="cuda")

    hf_output = hf_model.generate(
        **hf_inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    hf_text_output = [
        output.split("ASSISTANT:")[-1].strip()
        for output in processor.batch_decode(hf_output, skip_special_tokens=True)
    ]

    # cornstarch text generation
    cornstarch_inputs = copy.deepcopy(hf_inputs)
    cornstarch_output = cornstarch_model.generate(
        **cornstarch_inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    cornstarch_text_output = processor.batch_decode(
        cornstarch_output, skip_special_tokens=True
    )

    assert hf_text_output == cornstarch_text_output
