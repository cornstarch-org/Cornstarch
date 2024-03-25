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
    GPT2ForSequenceClassification,
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
    model_name_or_path: str = "gpt2"
    num_epoch: int = 3
    warmup_faction: float = 0.1


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    config: PretrainedConfig = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=GLUEDataBuilder.glue_task_num_labels["mrpc"]
    )
    config.pad_token_id = config.eos_token_id
    model = GPT2ForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )

    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    template1 = PipelineTemplate(
        model_name, [modules[:3], modules[3:8], modules[8:13], modules[13:]]
    )
    template2 = PipelineTemplate(model_name, [modules[:8], modules[8:]])
    plugin = HeterogeneousParallelPlugin(
        pipelines=[template2],
        tp_size=1,
        microbatch_size=4,
        num_microbatches={template2: 8},
        precision="bf16",
        enable_fused_normalization=True,
        enable_flash_attention=True,
    )
    booster = Booster(plugin=plugin)

    data_builder = GLUEDataBuilder(
        args.model_name_or_path, plugin, task_name="mrpc", pad_tokens=True
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
