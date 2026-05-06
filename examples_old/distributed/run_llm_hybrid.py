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
"""

import functools
from typing import Literal

import datasets
import torch
import torch.distributed as dist
import tyro
from colossalai.booster import Booster
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.shardformer.policies.auto_policy import _fullname
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

from cornstarch.shardformer.policies.auto_policy import get_autopolicy

model_names: dict[str, str] = {
    "gemma2": "google/gemma-2-2b-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2": "Qwen/Qwen2.5-3B-Instruct",
}


def tokenize_batch_for_pretrain(
    batch, tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    texts = [sample["text"] for sample in batch]
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    data = {k: v.to("cuda") for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


def pretrain(model_name: Literal["gemma2", "llama", "phi3", "mistral", "qwen2"]):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")

    model_path = model_names[model_name]

    # Get model from HF hub
    config = AutoConfig.from_pretrained(
        model_path, _attn_implementation="flash_attention_2"
    )

    with torch.device("meta"):
        model: PreTrainedModel = AutoModelForCausalLM.from_config(config).to(
            dtype=torch.bfloat16
        )
        model.train(mode=True)
        model.gradient_checkpointing_enable()

    # materialize the model
    with torch.no_grad():
        model.to_empty(device="cuda")
        for p in model.parameters():
            p.random_(0, 1)

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
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
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
    tyro.cli(pretrain)
