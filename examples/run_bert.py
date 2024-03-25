from dataclasses import dataclass
from pathlib import Path

import colossalai
import simple_parsing
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from data_builder import GLUEDataBuilder
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)

from oobleck_colossalai import (
    HeterogeneousDataLoader,
    HeterogeneousParallelPlugin,
    PipelineTemplate,
)


@dataclass
class ExampleArguments:
    model_name_or_path: str = "bert-base-uncased"
    num_epoch: int = 3
    warmup_faction: float = 0.1
    checkpoint_path: Path | None = None


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    config: PretrainedConfig = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=GLUEDataBuilder.glue_task_num_labels["mrpc"]
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    # Adjust module_per_stage to arbitrarily implement pipelines
    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    template1 = PipelineTemplate(model_name, [modules])
    template2 = PipelineTemplate(model_name, [modules[:8], modules[8:]])

    plugin = HeterogeneousParallelPlugin(
        pipelines=[template1, template2],
        tp_size=1,
        microbatch_size=4,
        num_microbatches={template1: 3, template2: 5},
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
    )
    booster = Booster(plugin=plugin)

    data_builder = GLUEDataBuilder(
        args.model_name_or_path, plugin, task_name="mrpc", pad_tokens=False
    )

    # Prepare dataloader
    dataloader: HeterogeneousDataLoader = data_builder.dataloader()

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
