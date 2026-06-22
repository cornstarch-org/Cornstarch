"""Microbatching through ``collate_fn`` returning a ``list[dict]``.

Microbatching is the user's responsibility: ``collate_fn`` returns the list of
microbatches for one optimizer step (Cornstarch never splits modality tensors
itself). ``prepare_dataloader`` yields that list, applying the DP/CP transforms
per microbatch. Without pipeline parallelism, consuming the list is plain
gradient accumulation — which must match processing the batch whole.
"""
from __future__ import annotations

import unittest

import torch
from torch.utils.data import Dataset

from tests.distributed.distributed_base import GlooDistributedTestBase
from tests.model.model_configs import llama_config

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    from_hf_config,
)

VOCAB = 128


class _TextDataset(Dataset):
    def __init__(self, length: int = 8, seq_len: int = 16) -> None:
        self.length = length
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        g = torch.Generator().manual_seed(index)
        ids = torch.randint(0, VOCAB, (self.seq_len,), generator=g)
        return {"input_ids": ids, "labels": ids.clone()}


def _split_collate(num_microbatches: int):
    """A user ``collate_fn`` returning a list of ``num_microbatches`` dicts."""

    def collate(samples: list) -> list[dict]:
        input_ids = torch.stack([s["input_ids"] for s in samples], dim=0)
        labels = torch.stack([s["labels"] for s in samples], dim=0)
        chunks_i = input_ids.chunk(num_microbatches, dim=0)
        chunks_l = labels.chunk(num_microbatches, dim=0)
        return [
            {"input_ids": ids, "labels": lbl}
            for ids, lbl in zip(chunks_i, chunks_l)
        ]

    return collate


def _build_model():
    config = llama_config()
    config.vocab_size = VOCAB
    config.tie_word_embeddings = False
    model = from_hf_config(config, model_kind="language", attn_implementation="eager")
    model.set_random_init()
    return model


def _lm_plan(model):
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
    )
    return plan.run_language_model(module=model, inputs=merged)


class TestMicrobatchDataloader(GlooDistributedTestBase):
    """``prepare_dataloader`` yields the user's microbatch list each iteration."""

    @property
    def world_size(self) -> int:
        return 1

    def test_yields_microbatch_list(self) -> None:
        model = _build_model()
        plan = ParallelizationPlan(global_ranks=[0])
        plan.parallelize(model, ParallelConfig(data_parallel_size=1))
        ctx = plan.materialize("cpu", dtype=torch.float32)

        loader = ctx.prepare_dataloader(
            _TextDataset(length=8, seq_len=16),
            batch_size=4,
            collate_fn=_split_collate(2),
        )
        microbatches = next(iter(loader))

        self.assertIsInstance(microbatches, list)
        self.assertEqual(len(microbatches), 2)
        for mb in microbatches:
            self.assertEqual(mb["input_ids"].shape, (2, 16))

    def test_bare_dict_collate_is_wrapped_as_single_microbatch(self) -> None:
        model = _build_model()
        plan = ParallelizationPlan(global_ranks=[0])
        plan.parallelize(model, ParallelConfig(data_parallel_size=1))
        ctx = plan.materialize("cpu", dtype=torch.float32)

        def dict_collate(samples: list) -> dict:
            return {
                "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0),
                "labels": torch.stack([s["labels"] for s in samples], dim=0),
            }

        loader = ctx.prepare_dataloader(
            _TextDataset(length=8, seq_len=16), batch_size=4, collate_fn=dict_collate
        )
        microbatches = next(iter(loader))
        self.assertIsInstance(microbatches, list)
        self.assertEqual(len(microbatches), 1)


class TestGradientAccumulation(GlooDistributedTestBase):
    """Non-PP microbatch consumption == gradient accumulation == whole batch."""

    @property
    def world_size(self) -> int:
        return 1

    def test_accumulation_matches_whole_batch(self) -> None:
        model = _build_model()
        plan = ParallelizationPlan(global_ranks=[0])
        plan.parallelize(model, ParallelConfig(data_parallel_size=1))
        ctx = plan.materialize("cpu", dtype=torch.float32)
        self.assertFalse(ctx.uses_pipeline_parallel)
        model.train()

        torch.manual_seed(0)
        batch = {
            "input_ids": torch.randint(0, VOCAB, (4, 16)),
            "labels": torch.randint(0, VOCAB, (4, 16)),
        }

        # Whole batch.
        output_future = _lm_plan(model)
        output_future.execute(inputs=batch).loss.backward()
        whole = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

        for p in model.parameters():
            p.grad = None

        # Two microbatches with scaled loss (gradient accumulation).
        num_microbatches = 2
        mbs = [
            {k: v.chunk(num_microbatches, dim=0)[i] for k, v in batch.items()}
            for i in range(num_microbatches)
        ]
        for mb in mbs:
            output_future = _lm_plan(model)
            loss = output_future.execute(inputs=mb).loss / num_microbatches
            loss.backward()
        accum = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

        self.assertEqual(set(whole), set(accum))
        for name in whole:
            torch.testing.assert_close(whole[name], accum[name], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
