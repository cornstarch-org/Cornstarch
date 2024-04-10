import functools
from dataclasses import dataclass
from pathlib import Path

import colossalai
import datasets
import simple_parsing
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
    LlamaForCausalLM,
    LlamaTokenizerFast,
    PretrainedConfig,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


@dataclass
class ExampleArguments:
    model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf"
    num_epoch: int = 3
    warmup_faction: float = 0.1
    checkpoint_path: Path | None = None


def tokenize_batch_for_pretrain(
    batch, tokenizer: PreTrainedTokenizer | None = None, max_length: int = 2048
):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


def main():
    args: ExampleArguments = simple_parsing.parse(ExampleArguments)

    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    config: PretrainedConfig = AutoConfig.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    model.gradient_checkpointing_enable()

    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    template_1stage = PipelineTemplate(model_name, [modules])
    # template_2stages = PipelineTemplate(model_name, [modules[:17], modules[17:]])
    # template_4stages = PipelineTemplate(
    #     model_name, [modules[:9], modules[9:18], modules[18:26], modules[26:]]
    # )
    plugin = HeterogeneousParallelPlugin(
        tp_size=4,
        microbatch_size=2,
        precision="bf16",
        enable_flash_attention=True,
    )
    plugin.set_pipelines(
        pipelines=[template_1stage], num_microbatches={template_1stage: 2}
    )

    booster = Booster(plugin=plugin)

    tokenizer = LlamaTokenizerFast.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    dataset = dataset["train"]
    dataloader: HeterogeneousDataLoader = plugin.prepare_dataloader(
        dataset,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(
            tokenize_batch_for_pretrain,
            tokenizer=tokenizer,
            max_length=model.config.max_position_embeddings,
        ),
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
