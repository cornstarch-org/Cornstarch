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

    # config vision feature layer to match cornstarch settings
    hf_model.config.vision_feature_layer = -1
    hf_model.config.vision_feature_select_strategy = "full"

    processor = AutoProcessor.from_pretrained(model_name)

    # loading sample image file
    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)

    # llava text generation
    prompt = "<image>USER: What are these? ASSISTANT:"
    hf_inputs = processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )

    # currently hf llava replaces <image> with actual image embeddings,
    # thus bos_token is located before image embedings, while cornstarch
    # simply prepends image embeddings to the input sequence,
    # thus the location of bos_token is different.
    # to match the result, we swap bos_token and <image>
    # FIXME: it does not guarantee correctness. Fix it.
    hf_inputs["input_ids"][0][0], hf_inputs["input_ids"][0][1] = (
        hf_inputs["input_ids"][0][1].clone(),
        hf_inputs["input_ids"][0][0].clone(),
    )

    hf_output = hf_model.generate(
        **hf_inputs,
        max_new_tokens=20,
        do_sample=False,
        vision_feature_layer=-1,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    hf_text_output = (
        processor.decode(hf_output[0][2:], skip_special_tokens=True)
        .split("ASSISTANT:")[-1]
        .strip()
    )

    # cornstarch text generation
    prompt = "USER: What are these? ASSISTANT:"
    cornstarch_inputs = processor(prompt, raw_image, return_tensors="pt").to(
        dtype=torch.float16, device="cuda"
    )
    cornstarch_output = cornstarch_model.generate(
        **cornstarch_inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    cornstarch_text_output = processor.decode(
        cornstarch_output[0], skip_special_tokens=True
    )

    assert hf_text_output == cornstarch_text_output
