"""
An example of pretraining a vision language model (VLM) using hybrid parallelism.

This still relies on colossalai APis for base features (e.g. Booster, DistCoordinate, etc),
Cornstarch is engaged in the following ways:
1.  Cornstarch provides `MultimodalModel` and related classes to create a vision language model
    from a vision model and a language model.
2.  Cornstarch provides `MultimodalParallelPlugin` and related classes to enable
    multimodal parallelism for the created vision language model.
3.  Cornstarch uses `PipelineTemplate` to specify how each stage in each modality
    should be sharded.

Basic flow is as follows:
1. Load models and tokenizers/processors using transformers `.from_pretrained()` method.
2. Create a `MultimodalModel` using the loaded models.
3. Create a `MultimodalModelProcessor` using the loaded tokenizers/processors.
---
Until here, it is the same with training non-distributed VLMs.
---
4. Create a `MultimodalParallelPlugin` using `ModalParallelPlugin` and `PipelineTemplate`.
5. Create ColossalAI `Booster` with the created plugin.
6. Configure the `MultinodalModel` with `MultimodalParallelPlugin` and `Booster`.
7. Use `Booster` to execute training.
"""

import functools
from pathlib import Path

import click
import colossalai
import torch
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from datasets import load_dataset
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.models.clip import CLIPImageProcessor, CLIPVisionModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalModelProcessor,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelPlugin,
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
        assert "<image>" in batch["conversations"][0]["value"]

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
def pretrain(
    vision_model_name_or_path: str,
    language_model_name_or_path: str,
    dataset_file_path: Path,
):
    colossalai.launch_from_torch()

    # Create a model
    if "clip" in vision_model_name_or_path:
        vision_encoder = CLIPVisionModel.from_pretrained(
            vision_model_name_or_path,
            trust_remote_code=False,
            _attn_implementation="eager",
        )
        vision_encoder = ModalEncoderModule(vision_encoder)
    else:
        raise NotImplementedError
    language_model = AutoModelForCausalLM.from_pretrained(
        language_model_name_or_path,
        trust_remote_code=False,
        _attn_implementation="eager",
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

    vision_encoder_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            model_name=PipelineTemplate.get_model_name(vision_encoder.module),
            modules_per_stage=[PipelineTemplate.get_modules(vision_encoder)],
        ),
    )
    language_model_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            model_name=PipelineTemplate.get_model_name(language_model),
            modules_per_stage=[PipelineTemplate.get_modules(language_model)],
        ),
    )
    num_microbatches = 4
    microbatch_size = 1
    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_encoder_plugin},
        language_model_plugin=language_model_plugin,
        num_microbatches=num_microbatches,
        microbatch_size=microbatch_size,
        precision=None,  # Don't use mixed precision and train with bf16
        enable_flash_attention=True,
    )
    booster = Booster(plugin=plugin)

    optimizer = Adam(model.parameters())
    optimizer.zero_grad()

    dataset = load_dataset("json", data_files=str(dataset_file_path))["train"]
    dataloader = plugin.prepare_dataloader(
        dataset,
        batch_size=microbatch_size * num_microbatches,
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
    is_pp_last_stage = plugin.stage_manager.is_last_stage(check_only_in_modal=False)

    dataloader_iter = iter(dataloader)
    with tqdm(
        range(total_steps),
        disable=not (coordinator.is_master() or is_pp_last_stage),
    ) as pbar:
        for item in pbar:
            outputs = booster.execute_pipeline(
                dataloader_iter,
                model,
                criterion,
                optimizer,
                return_loss=True,
                return_outputs=False,
            )

            if is_pp_last_stage:
                loss = outputs["loss"]
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    pretrain()
