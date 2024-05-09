import functools

import click
import colossalai
import datasets
import torch
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.optimizer import CPUAdam
from tqdm import tqdm
from transformers import PreTrainedTokenizer, get_linear_schedule_with_warmup
from transformers.models.llama import LlamaTokenizerFast
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.mixtral import MixtralForCausalLMPolicy


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


@click.command
@click.option(
    "--model_name_or_path",
    type=str,
    default="mistralai/Mixtral-8x7B-Instruct-v0.1",
    help="Model name or path.",
)
@click.option("--global_batch_size", type=int, default=1, help="Global batch size.")
def main(
    model_name_or_path: str,
    global_batch_size: int,
):
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()

    with LazyInitContext(default_device=get_accelerator().get_current_device()):
        config = MixtralConfig.from_pretrained(model_name_or_path)
        # TODO: remove it when example starts using pipeline parallelism
        config.num_hidden_layers = 4
        config._attn_implementation = "flash_attention_2"
        model = MixtralForCausalLM.from_pretrained(model_name_or_path, config=config)
    model.gradient_checkpointing_enable()

    model_name = PipelineTemplate.get_model_name(model)
    modules = PipelineTemplate.get_modules(model)
    pipeline_template = PipelineTemplate(model_name, [modules[:3], modules[3:]])

    policy = MixtralForCausalLMPolicy()
    policy.set_model(model)
    policy.set_pipeline_template(pipeline_template)

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    )
    tokenizer.pad_token = tokenizer.eos_token

    plugin = HybridParallelPlugin(
        tp_size=2,
        pp_size=2,
        microbatch_size=2,
        precision="bf16",
        custom_policy=policy,
    )
    dataloader = plugin.prepare_dataloader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=global_batch_size,
        collate_fn=functools.partial(tokenize_batch_for_pretrain, tokenizer=tokenizer),
    )

    booster = Booster(plugin=plugin)
    total_steps = len(dataloader)
    optimizer = CPUAdam(model.parameters())
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps,
    )

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )

    with torch.no_grad():
        for param in model.parameters():
            if param is None or param not in optimizer.working_to_master_map:
                continue
            master_param = optimizer.working_to_master_map[param]
            master_param.data = master_param.data.to(device="cpu")

    # Train model
    model.train()
    optimizer.zero_grad()
    is_pp_last_stage = plugin.stage_manager.is_last_stage()

    dataloader_iter = iter(dataloader)
    with tqdm(range(total_steps), disable=not (coordinator.is_master())) as pbar:
        for _ in pbar:
            outputs = booster.execute_pipeline(
                dataloader_iter,
                model,
                lambda outputs, inputs: outputs.loss,
                optimizer,
                return_loss=True,
                return_outputs=False,
            )

            if is_pp_last_stage:
                loss = outputs["loss"]
                pbar.set_postfix({"loss": loss.item()})

            for param in model.parameters():
                param.data = param.data.to(device="cpu")
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device="cpu")

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            for param in model.parameters():
                param.data = param.data.to(device="cuda")


if __name__ == "__main__":
    main()
