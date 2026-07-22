"""Acceptance tests for process-group-independent partition compilation."""

from __future__ import annotations

from unittest.mock import patch

from cornstarch.distributed import (
    ParallelConfig,
    ParallelizationPlan,
    PipelineStageSpec,
)
from cornstarch.models import from_hf_config
from tests.model.model_configs import llama_config


def _model_and_plan():
    model = from_hf_config(llama_config(), model_kind="language")
    plan = ParallelizationPlan(global_ranks=[0])
    plan.parallelize(model, ParallelConfig())
    return model, plan


def test_compile_does_not_touch_torch_distributed_or_allocate_storage():
    model, plan = _model_and_plan()
    layers = model._repeated_layers()
    spec = PipelineStageSpec("p", 0, 0, len(layers), (0,), True, True)
    with patch("torch.distributed.get_rank", side_effect=AssertionError("dist touched")):
        compiled = plan.compile(world_size=1, rank=0, stage_overrides=(spec,))
    assert compiled.rank == 0
    assert all(parameter.is_meta for parameter in model.parameters())


def test_compile_accepts_uneven_global_ranges_without_slicing_blueprint():
    model, plan = _model_and_plan()
    layers = model._repeated_layers()
    root_identity = id(model)
    blueprint_ids = tuple(map(id, layers))
    cut = max(1, len(layers) - 1)
    compiled = plan.compile(
        world_size=2,
        rank=0,
        stage_overrides=(PipelineStageSpec("p", 0, 0, cut, (0,), True, False),),
    )
    assert compiled.modules[0].stage.layer_end == cut
    assert id(model) == root_identity
    assert tuple(map(id, model._repeated_layers())) == blueprint_ids


def test_manifest_keys_follow_global_layer_when_ownership_moves_rank():
    model, plan = _model_and_plan()
    first = plan.compile(
        world_size=2,
        rank=0,
        stage_overrides=(PipelineStageSpec("a", 0, 0, 1, (0,), True, False),),
    )
    moved = plan.compile(
        world_size=2,
        rank=1,
        stage_overrides=(PipelineStageSpec("b", 1, 0, 1, (1,), False, True),),
    )
    assert {entry.logical_key for entry in first.local_state_manifest.entries} == {
        entry.logical_key for entry in moved.local_state_manifest.entries
    }


def test_activate_materializes_local_single_stage_and_close_is_idempotent():
    model, plan = _model_and_plan()
    compiled = plan.compile(world_size=1, rank=0)
    context = compiled.activate("cpu")
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert context.local_state_manifest is not None
    context.close()
    context.close()
    assert context.closed
    assert id(model) == id(compiled.modules[0].module)
