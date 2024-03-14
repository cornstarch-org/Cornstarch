from dataclasses import dataclass

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
    global_batch_size: int = 32
    num_epoch: int = 3
    warmup_faction: float = 0.1


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    plugin = HeterogeneousParallelPlugin(
        tp_size=1,
        global_batch_size=args.global_batch_size,
        microbatch_size=4,
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
    )
    booster = Booster(plugin=plugin)

    data_builder = GLUEDataBuilder(
        args.model_name_or_path, plugin, task_name="mrpc", pad_tokens=False
    )

    config: PretrainedConfig = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=data_builder.num_labels
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    # Adjust module_per_stage to arbitrarily implement pipelines
    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    template1 = PipelineTemplate(model_name, [modules])
    template2 = PipelineTemplate(model_name, [modules[:8], modules[8:]])
    plugin.set_pipeline_templates(
        # homogeneous pipelines with 4 GPUs
        pipeline_templates={template2: 2},
        num_microbatches={template2: 4},
        # heterogeneous pipelines with 3 GPUs
        # pipeline_templates={template1: 1, template2: 1},
        # num_microbatches={template1: 3, template2: 5},
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


if __name__ == "__main__":
    main()
