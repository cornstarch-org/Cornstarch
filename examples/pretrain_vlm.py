import functools
from pathlib import Path
from typing import Literal, Type

import torch
import tyro
from datasets import load_dataset
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.clip import CLIPVisionModel
from transformers.models.pixtral import PixtralVisionModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.siglip import SiglipVisionModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)

"""
Pretrain a vision language model with liuhaotian's llava-pretrain dataset:
https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
"""

model_names: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
    "pixtral": "mistral-community/pixtral-12b",
    "qwen2_vision": "Qwen/Qwen2-VL-2B-Instruct",
    "gemma2": "google/gemma-2-2b-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

vision_encoder_classes: dict[str, Type[PreTrainedModel]] = {
    "clip": CLIPVisionModel,
    "siglip": SiglipVisionModel,
    "pixtral": PixtralVisionModel,
    "qwen2_vision": Qwen2VisionTransformerPretrainedModel,
}


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
    ).to("cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def pretrain(
    dataset_file_path: Path,  # Path to the json file containing the dataset
    vision_encoder_name: Literal["clip", "siglip", "pixtral", "qwen2_vision"],
    llm_name: Literal["gemma2", "llama", "phi3", "mistral"],
):
    """
    Randomly initialize the model and pretrain it on the LLaVA-Pretrain dataset.
    """
    torch.cuda.set_device(0)

    vision_encoder_path = model_names[vision_encoder_name]
    llm_path = model_names[llm_name]

    print(f"Pretraining a VLM with {vision_encoder_path} + {llm_path}.")

    # Create a model
    with torch.device("meta"):
        vision_config = AutoConfig.from_pretrained(vision_encoder_path)
        vision_encoder = vision_encoder_classes[vision_encoder_name](
            vision_config.vision_config
        )

        llm_config = AutoConfig.from_pretrained(llm_path)
        language_model = AutoModelForCausalLM.from_config(llm_config)

        if vision_encoder_name == "qwen2_vision":
            # Qwen2 vision encoder is not designed as a standalone model,
            # but used in Qwen2VL, which has a preprocessing procedure for the input.
            # pixel_values -> hidden_states, image_grid_thw -> grid_thw
            vision_encoder = ModalEncoderModule(
                model=vision_encoder,
                additional_args=["pixel_values", "image_grid_thw"],
                preprocess_callback=lambda inputs: {
                    "hidden_states": inputs["pixel_values"],
                    "grid_thw": inputs["image_grid_thw"],
                },
            )
        else:
            vision_encoder = ModalEncoderModule(vision_encoder)

        model = MultimodalModel(
            encoders={"vision": vision_encoder},
            language_model=language_model,
        ).to(dtype=torch.bfloat16)

    # materialize the model
    with torch.no_grad():
        model.to_empty(device="cuda")
        for p in model.parameters():
            p.random_(0, 1)

    model.gradient_checkpointing_enable()
    model.train()

    # Create a processor
    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    processor = MultimodalProcessor(
        encoder_processors={"vision": image_processor},
        llm_tokenizer=tokenizer,
        model=model,
        predefined_tokens={"vision": "<image>"},
    )

    dataset = load_dataset("json", data_files=dataset_file_path.as_posix())["train"]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain,
            processor=processor,
            dataset_dir=dataset_file_path.parent,
        ),
    )

    optimizer = Adam([p for p in model.parameters() if p.requires_grad])
    optimizer.zero_grad()

    total_steps = len(dataloader)
    num_warmup_steps = int(total_steps * 0.1)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    dataloader_iter = iter(dataloader)
    with tqdm(
        range(total_steps),
    ) as pbar:
        for item in pbar:
            inputs = next(dataloader_iter)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
