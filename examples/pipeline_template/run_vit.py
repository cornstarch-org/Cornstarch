import functools
from dataclasses import dataclass
from pathlib import Path

import colossalai
import datasets
import simple_parsing
import torch
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from cornstarch import (
    HeterogeneousDataLoader,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoConfig,
    ViTConfig,
    ViTForImageClassification,
    ViTImageProcessor,
    get_linear_schedule_with_warmup,
)
from transformers.image_processing_utils import BaseImageProcessor


def transform(batch, image_processor: BaseImageProcessor):
    inputs = image_processor(batch["image"], return_tensors="pt")
    inputs["labels"] = batch["labels"]

    return inputs


def process_batch_for_pretrain(batch: tuple[torch.Tensor]):
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in batch]),
        "labels": torch.tensor(
            [sample["labels"] for sample in batch], dtype=torch.int64
        ),
    }


@dataclass
class ExampleArguments:
    model_name_or_path: str = "google/vit-base-patch16-224"
    num_epoch: int = 3
    warmup_faction: float = 0.1
    checkpoint_path: Path | None = None


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)
    tp_size = 2

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    dataset = datasets.load_dataset("beans")["train"]
    image_processor = ViTImageProcessor.from_pretrained(args.model_name_or_path)
    dataset = dataset.with_transform(
        functools.partial(transform, image_processor=image_processor)
    )

    config: ViTConfig = AutoConfig.from_pretrained(args.model_name_or_path)
    labels: list[str] = dataset.features["labels"].names
    while len(labels) % tp_size != 0:
        labels.append(f"pad_label_{len(labels)}")
    config.num_labels = len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}

    model = ViTForImageClassification.from_pretrained(
        args.model_name_or_path, config=config, ignore_mismatched_sizes=True
    )
    model.gradient_checkpointing_enable()

    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    template1 = PipelineTemplate(model_name, [modules])
    plugin = HeterogeneousParallelPlugin(
        tp_size=tp_size,
        microbatch_size=4,
        precision="bf16",
        enable_flash_attention=True,
    )
    plugin.set_pipelines(
        pipelines=[template1, template1], num_microbatches={template1: 8}
    )

    booster = Booster(plugin=plugin)

    dataloader: HeterogeneousDataLoader = plugin.prepare_dataloader(
        dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=process_batch_for_pretrain,
    )

    # optimizer
    optimizer = Adam(model.parameters())

    # lr_scheduler
    total_steps = len(dataloader) * args.num_epoch
    num_warmup_steps = int(total_steps * args.warmup_faction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model,
        optimizer=optimizer,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )

    if args.checkpoint_path and args.checkpoint_path.exists():
        booster.load_model(model, args.checkpoint_path / "model")
        booster.load_optimizer(optimizer, args.checkpoint_path / "optim")
        booster.load_lr_scheduler(
            lr_scheduler, args.checkpoint_path / "lr_scheduler.pt"
        )

    # Train model
    model.train()
    optimizer.zero_grad()
    is_pp_last_stage = plugin.stage_manager.is_last_stage()

    for epoch in range(args.num_epoch):
        total_step = len(dataloader)
        dataloader_iter = iter(dataloader)
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not (coordinator.is_master() or is_pp_last_stage),
        ) as pbar:
            for _ in pbar:
                outputs = booster.execute_pipeline(
                    dataloader_iter,
                    model,
                    criterion=lambda outputs, inputs: outputs.loss,
                    optimizer=optimizer,
                    return_loss=True,
                    return_outputs=True,
                )

                if is_pp_last_stage:
                    loss = outputs["loss"]
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

    if args.checkpoint_path:
        booster.save_model(
            model,
            checkpoint=args.checkpoint_path / "model",
            shard=True,
            use_safetensors=True,
        )
        booster.save_optimizer(
            optimizer, checkpoint=args.checkpoint_path / "optim", shard=True
        )
        booster.save_lr_scheduler(
            lr_scheduler, checkpoint=args.checkpoint_path / "lr_scheduler.pt"
        )


if __name__ == "__main__":
    main()
