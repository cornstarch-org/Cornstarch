"""Public plugin hooks for schedules and caller-owned batch planning."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch
from torch.utils.data import Dataset, SequentialSampler

from cornstarch.distributed.parallelization import ParallelContext, ScheduleContext


class _Dataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"value": torch.tensor(index)}


def _context(*, uses_pipeline_parallel: bool = False) -> ParallelContext:
    return ParallelContext(
        modules=[],
        configs=[],
        meshes={},
        layouts={},
        dp_size=1,
        dp_rank=0,
        dp_group=None,
        gradient_synchronizer=None,
        uses_pipeline_parallel=uses_pipeline_parallel,
    )


def test_prepare_dataloader_accepts_caller_sampler_and_transform() -> None:
    dataset = _Dataset()
    loader = _context().prepare_dataloader(
        dataset,
        batch_size=2,
        sampler=SequentialSampler(dataset),
        collate_fn=lambda samples: {
            "value": torch.stack([sample["value"] for sample in samples])
        },
        per_microbatch_transform=lambda batch: {
            **batch,
            "planned": torch.tensor(True),
        },
    )
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0]["value"].tolist() == [0, 1]
    assert batches[0][0]["planned"].item() is True


def test_prepare_dataloader_accepts_caller_batch_sampler() -> None:
    dataset = _Dataset()
    loader = _context().prepare_dataloader(
        dataset,
        batch_sampler=[[3, 1], [2, 0]],
        collate_fn=lambda samples: {
            "value": torch.stack([sample["value"] for sample in samples])
        },
    )
    assert [batch[0]["value"].tolist() for batch in loader] == [[3, 1], [2, 0]]


def test_schedule_factory_receives_immutable_public_context() -> None:
    captured = {}

    def factory(context: ScheduleContext):
        captured["context"] = context
        return "custom-schedule"

    result = _context(uses_pipeline_parallel=True).create_schedule(
        plan=object(),
        output_future=object(),
        schedule_factory=factory,
    )
    assert result == "custom-schedule"
    schedule_context = captured["context"]
    assert isinstance(schedule_context.layouts, MappingProxyType)
    with pytest.raises(TypeError):
        schedule_context.layouts[1] = object()


def test_prepare_dataloader_transforms_structured_encoder_and_llm_views() -> None:
    dataset = _Dataset()

    def collate(samples):
        values = torch.stack([sample["value"] for sample in samples]).unsqueeze(1)
        return {
            "encoder": {"input_ids": values.clone(), "labels": values.clone()},
            "language": {"input_ids": values.clone(), "labels": values.clone()},
            "plan_id": "iteration-0",
        }

    loader = _context().prepare_dataloader(
        dataset,
        batch_size=2,
        collate_fn=collate,
        microbatch_views=lambda planned: (
            planned["encoder"],
            planned["language"],
        ),
    )
    planned = next(iter(loader))[0]
    assert planned["plan_id"] == "iteration-0"
    assert "cp_global_input_ids" in planned["encoder"]
    assert "cp_global_input_ids" in planned["language"]
