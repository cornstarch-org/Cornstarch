from pathlib import Path
from typing import Type

import torch
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.clip import CLIPVisionModel
from transformers.models.pixtral import PixtralVisionModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.siglip import SiglipVisionModel

from cornstarch.models.multimodal_language_model import MultimodalProcessor


def collate_fn(batches: list[dict], processor: MultimodalProcessor):
    images = []
    texts = []

    for batch in batches:
        images.append(batch["image"])
        texts.append(batch["text"])

    inputs = processor(
        encoder_inputs={"vision": {"images": images}},
        llm_inputs={"text": texts, "padding": True},
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def collate_fn_llava_pretrain(
    batches: list[dict], processor: MultimodalProcessor, dataset_dir: Path
):
    images = []
    texts = []

    for batch in batches:
        for conversation in batch["conversations"]:
            assert ["from", "value"] == list(conversation.keys())

        if "image" in batch:
            image = Image.open(f"{dataset_dir}/{batch['image']}")
            image = image.convert("RGB")
            images.append(image)

        text = ""
        for conversation in batch["conversations"]:
            text += f"\"{conversation['from']}\"\n{conversation['value']}\n"

        texts.append(text)

    inputs = processor(
        encoder_inputs={"vision": {"images": images}} if images else None,
        llm_inputs={"text": texts, "padding": True},
        return_tensors="pt",
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


model_names: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
    "pixtral": "mistral-community/pixtral-12b",
    "qwen2_vision": "Qwen/Qwen2-VL-2B-Instruct",
    "gemma2": "google/gemma-2-2b-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2": "Qwen/Qwen2.5-3B-Instruct",
}

vision_encoder_classes: dict[str, Type[PreTrainedModel]] = {
    "clip": CLIPVisionModel,
    "siglip": SiglipVisionModel,
    "pixtral": PixtralVisionModel,
    "qwen2_vision": Qwen2VisionTransformerPretrainedModel,
}
