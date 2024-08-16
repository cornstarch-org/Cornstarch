"""
An example of pretraining a language model (LLM) using hybrid parallelism.

Most of the code is the same as the original pretraining script from colossalai.
The only difference is that the script uses Cornstarch's `get_autopolicy` to get a custom sharding policy
for the model that colossalai does not support.

Run:
torchrun (dist configs) run_llm_hybrid.py --model_name_or_path meta-llama/Meta-Llama-3-8B
For single-node multi-GPU training: torchrun --standalone --nproc-per-node=N
For multi-node training: torchrun --master-addr <ip> --master-port <port> --nproc-per-node=N
(need to run on all nodes)

Supported models: llama, gemma, gemma2, phi3, qwen2, mistral, mixtral
Example model
llama: meta-llama/Meta-Llama-3-8B
gemma: google/gemma-7b
gemma2: google/gemma-2-9b
phi3: microsoft/Phi-3-mini-4k-instruct
qwen2: Qwen/Qwen2-7B-Instruct
mistral: mistralai/Mistral-7B-Instruct-v0.3
mixtral: mistralai/Mixtral-8x7B-Instruct-v0.1
"""

import functools

import click
import colossalai
import datasets
import torch
from colossalai.booster import Booster
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.shardformer.policies.auto_policy import _fullname
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from cornstarch.shardformer.policies.auto_policy import get_autopolicy


def tokenize_batch_for_pretrain(batch, tokenizer: PreTrainedTokenizer):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


@click.command
@click.option(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Model name from HF hub or local path.",
    default="meta-llama/Meta-Llama-3-8B",
)
def pretrain(model_name_or_path: str):
    colossalai.launch_from_torch()

    # Get model from HF hub
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        _attn_implementation="eager",
    ).to(dtype=torch.bfloat16, device=torch.device("cuda"))
    model.train(mode=True)
    model.gradient_checkpointing_enable()

    # Get a custom sharding policy from Cornstarch.
    # Because colossalai HybridParallelPlugin finds the policy within its own policy list,
    # you need to manually pass the policy for the model that colossalai does not support.
    # You may use the corresponding ColossalAI's policy if your model is supported.
    policy = get_autopolicy(_fullname(model))

    # Create a plugin and a booster
    num_microbatches = 4
    microbatch_size = 1
    plugin = HybridParallelPlugin(
        tp_size=2,
        pp_size=2,
        num_microbatches=num_microbatches,
        microbatch_size=microbatch_size,
        custom_policy=policy,
        enable_flash_attention=True,
        parallel_output=False,
        enable_metadata_cache=False,
    )
    # To avoid mixed precision training, set precision to None
    plugin.precision = None
    booster = Booster(plugin=plugin)

    # tokenizer, dataset, and dataloader
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    dataloader: DataLoader = plugin.prepare_dataloader(
        dataset,
        batch_size=microbatch_size * num_microbatches,
        shuffle=True,
        drop_last=True,
        collate_fn=functools.partial(tokenize_batch_for_pretrain, tokenizer=tokenizer),
    )

    # optimizer and lr_scheduler
    optimizer = Adam(model.parameters())

    total_steps = len(dataloader)
    warmup_fraction = 0.1
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # Parallelize the model and prepare training
    model, optimizer, criterion, dataloader, lr_scheduler = booster.boost(
        model,
        optimizer=optimizer,
        criterion=lambda outputs, inputs: outputs.loss,
        dataloader=dataloader,
        lr_scheduler=lr_scheduler,
    )

    optimizer.zero_grad()
    is_pp_last_stage = plugin.stage_manager.is_last_stage()

    dataloader_iter = iter(dataloader)
    coordinator = DistCoordinator()
    with tqdm(
        range(total_steps),
        disable=not (coordinator.is_master() or is_pp_last_stage),
    ) as pbar:
        for _ in pbar:
            outputs = booster.execute_pipeline(
                dataloader_iter,
                model,
                criterion=criterion,
                optimizer=optimizer,
                return_loss=True,
                return_outputs=False,
            )

            if is_pp_last_stage:
                loss = outputs["loss"]
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


if __name__ == "__main__":
    pretrain()
