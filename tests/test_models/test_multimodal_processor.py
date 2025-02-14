import io

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.siglip.modeling_siglip import SiglipVisionModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)


def create_random_image(width: int, height: int) -> Image:
    """
    Creates a fake JPEG image with random noise of specified width and height.

    Parameters:
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    PIL.JpegImagePlugin.JpegImageFile: An instance of a JPEG image with random noise.
    """
    # Generate random noise for the image (values between 0 and 255 for RGB channels)
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create an image from the random data
    image = Image.fromarray(random_data, "RGB")

    # Save the image to a BytesIO object to simulate a JPEG file
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Open the image from the BytesIO object as a JpegImageFile instance
    jpeg_image = Image.open(image_bytes)

    return jpeg_image


encoder_processors_dict = {
    "clip": (CLIPVisionModel, "llava-hf/llava-1.5-7b-hf"),
    "siglip": (SiglipVisionModel, "llava-hf/llava-1.5-7b-hf"),
    "qwen2": (Qwen2VisionTransformerPretrainedModel, "Qwen/Qwen2-VL-2B-Instruct"),
    "pixtral": (PixtralVisionModel, "mistral-community/pixtral-12b"),
}

llm_models_dict = {
    "gpt": "openai-community/gpt2",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "qwen2": "Qwen/Qwen2.5-7B-Instruct",
}


@pytest.mark.parametrize(
    "encoder_class_baseline_name",
    list(encoder_processors_dict.values()),
    ids=list(encoder_processors_dict.keys()),
)
@pytest.mark.parametrize(
    "llm_model_name",
    list(llm_models_dict.values()),
    ids=list(llm_models_dict.keys()),
)
def test_multimodal_processor_generate_image_tokens(
    encoder_class_baseline_name: tuple[PreTrainedModel, str],
    llm_model_name: str,
):
    encoder_class, baseline_name = encoder_class_baseline_name

    with torch.device("meta"):
        baseline_config = AutoConfig.from_pretrained(baseline_name)

        encoder_module = encoder_class(baseline_config.vision_config)
        llm = AutoModel.from_config(
            baseline_config.text_config
            if hasattr(baseline_config, "text_config")
            else baseline_config
        )

        model = MultimodalModel(
            {
                "vision": ModalEncoderModule(encoder_module),
            },
            language_model=llm,
        )

    baseline_processor = AutoProcessor.from_pretrained(baseline_name)
    baseline_token = baseline_processor.image_token

    mm_processor = MultimodalProcessor(
        {"vision": baseline_processor.image_processor},
        llm_tokenizer=AutoTokenizer.from_pretrained(llm_model_name),
        model=model,
        predefined_tokens={"vision": baseline_token},
    )

    image1 = create_random_image(336, 336)
    image2 = create_random_image(480, 320)

    text = f"Image 1: {baseline_token}. Image 2:"
    inputs = mm_processor(
        encoder_inputs={
            "vision": {"images": [image1, image2]},
        },
        llm_inputs={"text": text},
        return_tensors="pt",
    )

    baseline_inputs = baseline_processor(
        images=[image1, image2], text=text, return_tensors="pt"
    )

    assert (
        inputs["input_ids"]
        == mm_processor.llm_tokenizer.convert_tokens_to_ids(baseline_token)
    ).sum() == (
        baseline_inputs["input_ids"]
        == baseline_processor.tokenizer.convert_tokens_to_ids(baseline_token)
    ).sum()
