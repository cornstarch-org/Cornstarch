from __future__ import annotations

from types import SimpleNamespace

import pytest

from cornstarch.distributed.pipeline_parallel import (
    PipelinePartitionSpec,
    apply_pipeline_parallel,
)
from cornstarch.models import from_hf_config
from tests.model.model_configs import llama_config


def test_partition_spec_validates_nonempty_contiguous_exhaustive_ranges() -> None:
    spec = PipelinePartitionSpec((1, 3, 6))
    assert spec.layer_range(total_layers=6, stage=0, num_stages=3) == (0, 1)
    assert spec.layer_range(total_layers=6, stage=1, num_stages=3) == (1, 3)
    assert spec.layer_range(total_layers=6, stage=2, num_stages=3) == (3, 6)

    with pytest.raises(ValueError, match="exhaustive"):
        spec.layer_range(total_layers=7, stage=0, num_stages=3)
    with pytest.raises(ValueError, match="nonempty"):
        PipelinePartitionSpec((1, 1, 3))


def test_apply_pipeline_parallel_consumes_explicit_boundaries() -> None:
    model = from_hf_config(llama_config(), model_kind="language")
    total = len(model.decoder_layers)
    assert total >= 2
    spec = PipelinePartitionSpec((1, total))
    mesh = SimpleNamespace(stage=0, num_stages=2, distribute_layers=lambda _: (0, total // 2))

    apply_pipeline_parallel(model, mesh, partition=spec)

    assert len(model.decoder_layers) == 1
    assert model._pipeline_layer_offset == 0
