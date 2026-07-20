"""Shared scaffolding for the distributed pretraining examples.

The public plan/context surface keeps rank math, the
TP->CP->PP-then-materialize-then-EP ordering rule, sampler/splitter wiring, and
gradient synchronization inside
``ParallelizationPlan`` / ``ParallelContext``.  What is left here is only the
shared, non-parallelism scaffolding (NCCL bring-up, device transfer, a
synthetic dataset, and the criterion).  Each ``pretrain_*`` script then reads
top-to-bottom like ``examples/pretrain_vlm.py``: build the models, parallelize,
materialize, and drive an explicit training loop with ``schedule.step``.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

DTYPE = torch.bfloat16


def init_distributed() -> tuple[int, int, torch.device]:
    """Bind this torchrun process to its local GPU and initialize NCCL.

    ``LOCAL_RANK`` is node-local, unlike the global rank, so this works for both
    single-node and multi-node launches. Cornstarch's correctness tests use
    gloo where useful; training examples intentionally model the production
    one-process-per-GPU NCCL setup.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training examples require CUDA and NCCL.")
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but this node exposes only "
            f"{torch.cuda.device_count()} CUDA devices."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return dist.get_rank(), dist.get_world_size(), device


def move_microbatches_to_device(
    microbatches: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    """Transfer model inputs after DP sampling and CP collation are complete."""
    return [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in microbatch.items()
        }
        for microbatch in microbatches
    ]


def local_trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    """Return concrete parameters owned by this rank, excluding remote meta roots."""
    return [
        parameter
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad and not parameter.is_meta
    ]


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


def context_parallel_language_inputs(enabled: bool) -> dict[str, Any]:
    """Return the data-side CP metadata futures consumed by the LM merge node."""
    if not enabled:
        return {}
    from cornstarch.models import ExecutionFuture

    return {
        key: ExecutionFuture(key)
        for key in (
            "attention_mask",
            "position_ids",
            "shift_labels",
            "num_items_in_batch",
            "cp_global_input_ids",
        )
    }


def microbatch_collate(
    num_microbatches: int,
    collate_fn: Callable[[list], dict] | None = None,
) -> Callable[[list], list[dict]]:
    """Build a collator returning exactly ``num_microbatches`` equal batches.

    Microbatching is the user's responsibility (Cornstarch never splits modality
    tensors itself), and the chosen interface is "``collate_fn`` returns a list of
    microbatches". This helper covers the simple case where every per-sample value
    can be split along the sample (dim 0) axis — which holds for the synthetic
    text and one-image-per-sample datasets here. A real VLM with a varying number
    of images per sample would write its own ``collate_fn`` that keeps each
    sample's pixels with its text while still returning a ``list[dict]``.

    ``ctx.prepare_dataloader`` applies the DP-sampler / CP-split transforms to each
    returned microbatch independently.
    """

    if num_microbatches < 1:
        raise ValueError("num_microbatches must be positive.")

    def collate(samples: list) -> list[dict]:
        batch = collate_fn(samples) if collate_fn is not None else _default_collate(samples)
        n = len(samples)
        if n == 0 or n % num_microbatches:
            raise ValueError(
                f"Batch size {n} must be nonzero and divisible by "
                f"num_microbatches={num_microbatches}."
            )
        per = n // num_microbatches
        microbatches: list[dict] = []
        for start in range(0, n, per):
            stop = start + per
            microbatches.append(
                {
                    key: value[start:stop] if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
            )
        return microbatches

    return collate


def _default_collate(samples: list) -> dict:
    """Stack a list of dict samples into a batch dict (tensors stacked dim 0)."""
    keys = samples[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [s[key] for s in samples]
        batch[key] = (
            torch.stack(values, dim=0)
            if isinstance(values[0], torch.Tensor)
            else torch.tensor(values)
        )
    return batch
