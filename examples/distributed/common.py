"""Shared scaffolding for the Option C distributed pretraining examples.

The Option C surface keeps the per-script boilerplate the earlier trial
duplicated — rank math, the TP->CP->PP-then-materialize-then-EP ordering rule,
sampler/splitter wiring, the grad-sync registration — inside
``ParallelizationPlan`` / ``ParallelContext``.  What is left here is only the
genuinely shared, non-parallelism scaffolding (process-group bring-up, a
synthetic dataset, and the criterion).  Each ``pretrain_*`` script then reads
top-to-bottom like ``examples/pretrain_vlm.py``: build the models, parallelize,
materialize, and drive an explicit training loop with ``schedule.step``.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(index)
        ids = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
        return {"input_ids": ids, "labels": ids.clone()}


def causal_lm_criterion(output: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the loss from a model output or a raw loss tensor (PP last stage)."""
    if isinstance(output, torch.Tensor):
        return output
    return output.loss if hasattr(output, "loss") else output["loss"]
