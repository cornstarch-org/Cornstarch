import functools
from pathlib import Path

import click
import torch
from datasets import load_dataset
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    LlamaTokenizerFast,
    get_linear_schedule_with_warmup,
)

from cornstarch.models.multimodal_language_model import (
    MultimodalLanguageModel,
    MultimodalLanguageModelProcessor,
)


def collate_fn_llava_pretrain(
    batches: list[dict], processor: MultimodalLanguageModelProcessor, dataset_dir: Path
):
    images = []
    texts = []

    for batch in batches:
        assert ["image", "id", "conversations"] == list(batch.keys())
        assert (
            isinstance(batch["conversations"], list)
            and len(batch["conversations"]) == 2
        )
        for conversation in batch["conversations"]:
            assert ["from", "value"] == list(conversation.keys())
        assert "<image>" in batch["conversations"][0]["value"]

        image = Image.open(f"{dataset_dir}/{batch['image']}")
        images.append(image)

        texts.append(f"<image> {batch['conversations'][1]['value']}")

    inputs = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
    )
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to("cuda").requires_grad_(v.is_floating_point())
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


@click.command
@click.option("--dataset_dir", type=Path, required=True, help="Path to the dataset.")
@click.option(
    "--dataset_file_name", type=str, required=True, help="Name of main dataset."
)
@click.option("--num_epoch", type=int, default=3, help="Number of epochs.")
@click.option("--warmup_fraction", type=float, default=0.1, help="Warmup fraction.")
def pretrain(
    dataset_dir: Path, dataset_file_name: str, num_epoch: int, warmup_fraction: float
):
    torch.cuda.set_device(0)

    # Create a model
    model: MultimodalLanguageModel = (
        MultimodalLanguageModel.from_encoders_llm_pretrained(
            text_model_name_or_path="microsoft/Phi-3-mini-128k-instruct",
            vision_model_name_or_path="openai/clip-vit-base-patch32",
        ).to(dtype=torch.bfloat16, device="cuda")
    )
    model.gradient_checkpointing_enable()

    # Create a processor
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    text_processor = LlamaTokenizerFast.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
    )
    processor = MultimodalLanguageModelProcessor(
        tokenizer=text_processor,
        image_processor=image_processor,
    )
    # Must resize embedding otherwise embedding will experience out of index error
    model.resize_token_embeddings(len(processor.tokenizer))

    """
    Examples of loading some datasets:
    1. liuhaotian/llava-pretrain
        tree -L 1 /path/to/datasets/liuhaotian___llava-pretrain
        liuhaotian___llava-pretrain/
        |-- 00000
        |-- 00001
        |-- 00002
        ...
        |-- 00658
        |-- 00659
        |-- blip_laion_cc_sbu_558k.json

        dataset_dir = /path/to/datasets/liuhaotian___llava-pretrain
        dataset_file_name = blip_laion_cc_sbu_558k.json
    """

    dataset = load_dataset("json", data_files=f"{dataset_dir}/{dataset_file_name}")[
        "train"
    ]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain, processor=processor, dataset_dir=dataset_dir
        ),
    )

    model = model.train(
        train_language_model="frozen",
        train_vision_model="frozen",
        train_projection="full",
    )
    processor.train()

    optimizer = Adam(model.parameters())
    optimizer.zero_grad()
    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    for epoch in range(num_epoch):
        total_step = len(dataloader)
        dataload_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                pbar.set_postfix({"loss": loss.item()})

    print("Saving projection module...")
    torch.save(model.vision_model.projection, "clip_phi3_vision_projection.pth")


if __name__ == "__main__":
    pretrain()
