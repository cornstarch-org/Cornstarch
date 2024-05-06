import functools
from pathlib import Path

import click
import colossalai
import torch
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.nn.optimizer import HybridAdam
from datasets import load_dataset
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    CLIPImageProcessor,
    get_linear_schedule_with_warmup,
)
from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast

from cornstarch.models.evaclip import EvaCLIPVisionModel
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
            inputs[k] = v.to(device="cuda")
        if v.is_floating_point():
            inputs[k] = inputs[k].requires_grad_(True)
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
    colossalai.launch_from_torch({})

    # Create a model
    model: MultimodalLanguageModel = (
        MultimodalLanguageModel.from_encoders_llm_pretrained(
            text_model_name_or_path="google/gemma-1.1-2b-it",
            vision_model_name_or_path="BAAI/EVA-CLIP-8B",
            vision_model_cls=EvaCLIPVisionModel,
        ).to(dtype=torch.bfloat16)
    )
    model.gradient_checkpointing_enable()

    # Create a processor
    # ImageProcessor configuration from https://huggingface.co/BAAI/EVA-CLIP-8B
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    text_processor = GemmaTokenizerFast.from_pretrained(
        "google/gemma-1.1-2b-it",
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

    plugin = GeminiPlugin(precision="bf16", offload_optim_frac=1.0)
    booster = Booster(plugin=plugin)

    dataset = load_dataset("json", data_files=f"{dataset_dir}/{dataset_file_name}")[
        "train"
    ]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
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

    optimizer = HybridAdam(model.parameters())
    optimizer.zero_grad()

    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model,
        optimizer=optimizer,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )

    for epoch in range(num_epoch):
        total_step = len(dataloader)
        dataload_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
            disable=torch.distributed.get_rank() != 0,
        ) as pbar:
            for i, item in enumerate(pbar):
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                loss = outputs.loss
                booster.backward(loss, optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                pbar.set_postfix({"loss": loss.item()})

                if i == 5:
                    return

    print("Saving projection module...")
    torch.save(
        model.unwrap().vision_model.projection,
        "evaclip_gemma_vision_projection_fsdp.pth",
    )


if __name__ == "__main__":
    pretrain()
