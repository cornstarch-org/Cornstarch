import copy
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
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
)

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)

model_name_classes = [
    ("llava-hf/llava-1.5-7b-hf", LlavaForConditionalGeneration),
    ("llava-hf/llava-v1.6-vicuna-7b-hf", LlavaNextForConditionalGeneration),
]


@pytest.mark.parametrize(
    "model_name_class", model_name_classes, ids=lambda x: x[0].split("/")[1]
)
def test_multimodal_model_generation(
    model_name_class: tuple[str, Type[PreTrainedModel]],
):
    model_name, model_cls = model_name_class

    # TODO: currently does not support llava-v1.6 due to its image patching
    if "llava-v1.6" in model_name:
        pytest.skip("llava-v1.6 not supported")

    # create cornstarch llava model
    cornstarch_model: MultimodalModel = (
        MultimodalModel.from_pretrained_multimodal_model(
            pretrained_model_id=model_name,
        ).to(dtype=torch.float16, device="cuda")
    )
    cornstarch_model.train(mode=False)

    hf_model: PreTrainedModel = model_cls.from_pretrained(model_name).to(
        dtype=torch.float16, device="cuda"
    )
    hf_model.train(mode=False)

    processor = AutoProcessor.from_pretrained(model_name)

    # loading sample image file
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # llava text generation
    prompt = "<image>USER: What are these? ASSISTANT:"
    hf_inputs = processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )

    hf_output = hf_model.generate(
        **hf_inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    hf_text_output = (
        processor.decode(hf_output[0][2:], skip_special_tokens=True)
        .split("ASSISTANT:")[-1]
        .strip()
    )

    # cornstarch text generation
    cornstarch_inputs = copy.deepcopy(hf_inputs)
    cornstarch_output = cornstarch_model.generate(
        **cornstarch_inputs,
        max_new_tokens=20,
        do_sample=False,
    )
    cornstarch_text_output = processor.decode(
        cornstarch_output[0], skip_special_tokens=True
    )

    assert hf_text_output == cornstarch_text_output
