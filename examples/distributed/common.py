"""Shared helpers for the Option C distributed pretraining examples.

The whole point of the Option C surface is that the per-script boilerplate the
earlier trial duplicated — rank math, the TP->CP->PP-then-materialize-then-EP
ordering rule, sampler/splitter wiring, the grad-sync registration — now lives
inside ``ParallelizationPlan`` / ``ParallelContext``.  These helpers only cover
the genuinely shared scaffolding (process-group bring-up, a synthetic dataset,
the criterion, and the training loop), so each ``pretrain_*`` script is just a
declarative description of *which* modality gets *which* degrees.
"""
from __future__ import annotations

import os
from typing import Callable

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from cornstarch.distributed import ParallelContext
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)

DTYPE = torch.bfloat16


def init_distributed() -> tuple[int, int, torch.device]:
    """Initialize the default process group from torchrun env vars.

    Returns ``(rank, world_size, device)``.  Uses NCCL on CUDA and gloo on CPU
    so the same script runs in both.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    return rank, world_size, device


class FakeTextDataset(Dataset):
    """Tiny synthetic causal-LM dataset of random token ids."""

    def __init__(self, vocab_size: int, seq_len: int, length: int = 4096) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(index)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
        return {"input_ids": ids, "labels": ids.clone()}


def build_language_model(model_name_or_path: str, attn_implementation: str = "sdpa"):
    """Build a meta-device Cornstarch language model with a random init plan."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name_or_path)
    config = getattr(config, "text_config", config)
    model = from_hf_config(
        config, model_kind="language", attn_implementation=attn_implementation
    )
    model.set_random_init()
    return model


def causal_lm_criterion(output, batch) -> torch.Tensor:
    """Return the loss from a model output or a raw loss tensor (PP last stage)."""
    if isinstance(output, torch.Tensor):
        return output
    return output.loss if hasattr(output, "loss") else output["loss"]


def build_language_model_plan(model) -> tuple[CornstarchExecutionPlan, ExecutionFuture]:
    """Build the LM-only execution DAG (no modality encoders)."""
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
    )
    output_future = plan.run_language_model(module=model, inputs=merged)
    return plan, output_future


def train_loop(
    ctx: ParallelContext,
    schedule,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    steps: int,
) -> None:
    """Run a plain PyTorch training loop over the Option C surface.

    The only parallelism-aware calls are ``ctx.sync_gradients()`` after the
    schedule step (DP all-reduce, EP shards skipped) and using the schedule's
    ``step`` for forward+backward (which microbatches under PP).
    """
    step = 0
    for batch in loader:
        if step >= steps:
            break
        result = schedule.step(batch, criterion, optimizer, return_loss=True)
        ctx.sync_gradients()
        optimizer.step()
        optimizer.zero_grad()
        if result["loss"] is not None and dist.get_rank() == 0:
            print(f"step {step}: loss {result['loss'].item():.4f}")
        step += 1
