"""
An example of pretraining a vision language model (VLM) using Pytorch FullyShardedDataParallel (FSDP).

This relies on existing colossalai APIs (TorchFSDPPlugin).
Cornstarch is used only for generating a `MultimodalModel`.
"""

import functools
from pathlib import Path

import click
import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.booster.plugin.torch_fsdp_plugin import TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from datasets import load_dataset
from PIL import Image
from torch.distributed.fsdp.api import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)
from transformers.models.clip import CLIPImageProcessor
from transformers.models.clip.modeling_clip import (
    CLIPEncoderLayer,
    CLIPVisionModel,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
)

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalModelProcessor,
    MultimodalProjector,
)


def collate_fn_llava_pretrain(
    batches: list[dict], processor: MultimodalModelProcessor, dataset_dir: Path
):
    images = []
    texts = []

    for batch in batches:
        assert set(["image", "id", "conversations"]) == set(batch.keys())
        assert (
            isinstance(batch["conversations"], list)
            and len(batch["conversations"]) == 2
        )
        for conversation in batch["conversations"]:
            assert ["from", "value"] == list(conversation.keys())
        # assert "<image>" in batch["conversations"][0]["value"]
        batch["conversations"][0]["value"].replace("<image>", "")

        image = Image.open(f"{dataset_dir}/{batch['image']}")
        images.append(image)

        texts.append(
            f"{batch['conversations'][0]['value']} "
            f"{batch['conversations'][1]['value']}"
        )

    data = processor(images=images, text=texts, return_tensors="pt", padding=True)
    data = {k: v.to("cuda") for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


@click.command
@click.option(
    "--vision_model_name_or_path",
    type=str,
    required=True,
    help="Vision model name from HF hub or local path.",
    default="openai/clip-vit-base-patch32",
)
@click.option(
    "--language_model_name_or_path",
    type=str,
    required=True,
    help="Language model name from HF hub or local path.",
    default="meta-llama/Meta-Llama-3-8B",
)
@click.option(
    "--dataset_file_path", type=Path, required=True, help="Path to the main dataset."
)
@click.option("--output_dir", type=Path, required=True, help="Path to save the model.")
def pretrain(
    vision_model_name_or_path: str,
    language_model_name_or_path: str,
    dataset_file_path: Path,
    output_dir: Path,
):
    colossalai.launch_from_torch()

    # Create a model
    if "clip" in vision_model_name_or_path:
        vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name_or_path,
            trust_remote_code=False,
            _attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        vision_encoder = ModalEncoderModule(vision_encoder)
    else:
        raise NotImplementedError
    language_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        language_model_name_or_path,
        trust_remote_code=False,
        _attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    model = MultimodalModel(
        encoders={"vision": vision_encoder},
        language_model=language_model,
    ).to(dtype=torch.bfloat16, device="cuda")
    vision_encoder.train(module=False, projector=True)
    language_model.train(mode=False)
    model.gradient_checkpointing_enable()

    # Create a processor
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name_or_path)
    text_processor = AutoTokenizer.from_pretrained(
        language_model_name_or_path, use_fast=True
    )
    processor = MultimodalModelProcessor(
        tokenizer=text_processor,
        image_processor=image_processor,
    )

    # This pad token will be used in callback to replace image token
    language_model.config.pad_token_id = text_processor.pad_token_id

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

    plugin = TorchFSDPPlugin(
        process_group=dist.group.WORLD,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=ModuleWrapPolicy(
            [
                MultimodalProjector,
                CLIPEncoderLayer,
                LlamaDecoderLayer,
                nn.Embedding,
            ]
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    plugin.fsdp_kwargs["forward_prefetch"] = True

    microbatch_size = 16

    booster = Booster(plugin=plugin)

    optimizer = Adam(model.parameters())
    optimizer.zero_grad()

    dataset = load_dataset("json", data_files=str(dataset_file_path))["train"]
    dataloader = plugin.prepare_dataloader(
        dataset,
        batch_size=microbatch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain,
            processor=processor,
            dataset_dir=dataset_file_path.parent,
        ),
    )

    total_steps = len(dataloader)
    warmup_fraction = 0.1
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, criterion, *_ = booster.boost(
        model=model,
        optimizer=optimizer,
        criterion=lambda outputs, inputs: outputs.loss,
    )

    coordinator = DistCoordinator()
    optimizer.zero_grad()

    dataloader_iter = iter(dataloader)
    with tqdm(
        range(total_steps),
        disable=not (coordinator.is_master()),
    ) as pbar:
        for item in pbar:
            inputs = next(dataloader_iter)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            if coordinator.is_master():
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    booster.save_model(
        model,
        output_dir / f"{vision_model_name_or_path}-{language_model_name_or_path}",
        shard=True,
        use_safetensors=True,
    )


if __name__ == "__main__":
    pretrain()
