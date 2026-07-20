"""Unit tests for the ``ParallelizationPlan`` surface that need no process group.

These exercise the declarative plan's input validation — specifically the
misconfiguration guard that rejects a non-Cornstarch module (e.g. a bare HF
encoder) before any distributed setup happens.
"""
from __future__ import annotations

import pytest
import torch.nn as nn

from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import build_modality_encoder, from_hf_config
from tests.model.model_configs import clip_vision_config, llama_config


def test_parallelize_rejects_non_cornstarch_module() -> None:
    """A bare ``nn.Module`` is rejected with a wrap-it hint, not silently accepted."""
    plan = ParallelizationPlan(global_ranks=[0])
    with pytest.raises(TypeError, match="build_modality_encoder"):
        plan.parallelize(nn.Linear(4, 4), ParallelConfig())


def test_parallelize_accepts_cornstarch_model_and_modality_encoder() -> None:
    """Both a Cornstarch model and a modality encoder register without error."""
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )

    plan = ParallelizationPlan(global_ranks=[0])
    plan.parallelize(language_model, ParallelConfig())
    plan.parallelize(modality_encoder, ParallelConfig())


def test_plan_requires_every_world_rank_exactly_once(monkeypatch) -> None:
    """A module grid cannot silently strand or duplicate a distributed rank."""
    monkeypatch.setattr(
        "cornstarch.distributed.parallelization.dist.get_world_size", lambda: 4
    )

    with pytest.raises(ValueError, match="every world rank exactly once"):
        ParallelizationPlan(global_ranks=[0, 1, 1, 3])._resolve_global_ranks()
    with pytest.raises(ValueError, match="every world rank exactly once"):
        ParallelizationPlan(global_ranks=[0, 1, 2])._resolve_global_ranks()

    assert ParallelizationPlan(
        global_ranks=[3, 1, 0, 2]
    )._resolve_global_ranks() == [3, 1, 0, 2]


def test_parallel_config_pipeline_parallel_size_defaults_to_none() -> None:
    """``pipeline_parallel_size`` defaults to None (co-locate / no PP)."""
    config = ParallelConfig()
    assert config.pipeline_parallel_size is None
    assert config.uses_pipeline_parallel is False
    assert config.num_pp_stages == 1  # None counts as a single stage for rank math
    assert config.ranks_per_replica == 1


def test_parallel_config_positive_pipeline_parallel_size_is_pp() -> None:
    """A positive ``pipeline_parallel_size`` means pipeline parallelism is used."""
    config = ParallelConfig(tensor_parallel_size=2, pipeline_parallel_size=2)
    assert config.uses_pipeline_parallel is True
    assert config.num_pp_stages == 2
    assert config.ranks_per_replica == 4


def test_parallel_config_rejects_non_positive_pipeline_parallel_size() -> None:
    """``pipeline_parallel_size`` is either None or a positive int."""
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        ParallelConfig(pipeline_parallel_size=0)


def test_assign_ranks_colocates_when_no_pipeline_parallel() -> None:
    """All-None ``pipeline_parallel_size`` co-locates every module on shared ranks."""
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )
    plan = ParallelizationPlan(global_ranks=[0, 1])
    plan.parallelize(modality_encoder, ParallelConfig(data_parallel_size=2))
    plan.parallelize(language_model, ParallelConfig(data_parallel_size=2))

    pipelined, dp_size, module_ranks = plan._assign_ranks([0, 1], 2)
    assert pipelined is False
    assert dp_size == 2
    assert module_ranks == [[0, 1], [0, 1]]  # both modules span all ranks


def test_assign_ranks_disaggregates_when_pipeline_parallel() -> None:
    """All-positive ``pipeline_parallel_size`` disaggregates onto disjoint ranks."""
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )
    plan = ParallelizationPlan(global_ranks=[0, 1])
    plan.parallelize(
        modality_encoder, ParallelConfig(pipeline_parallel_size=1, data_parallel_size=1)
    )
    plan.parallelize(
        language_model, ParallelConfig(pipeline_parallel_size=1, data_parallel_size=1)
    )

    pipelined, dp_size, module_ranks = plan._assign_ranks([0, 1], 2)
    assert pipelined is True
    assert dp_size == 1
    assert module_ranks == [[0], [1]]  # disjoint


def test_assign_ranks_rejects_mixed_pipeline_intent() -> None:
    """A mix of None and positive ``pipeline_parallel_size`` is rejected."""
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )
    plan = ParallelizationPlan(global_ranks=[0, 1])
    plan.parallelize(modality_encoder, ParallelConfig(data_parallel_size=1))
    plan.parallelize(
        language_model, ParallelConfig(pipeline_parallel_size=1, data_parallel_size=1)
    )
    with pytest.raises(ValueError, match="agree on pipeline parallelism"):
        plan._assign_ranks([0, 1], 2)


def test_assign_ranks_rejects_unequal_colocated_ranks_per_replica() -> None:
    """Co-located modules must have equal ranks_per_replica (tp*cp*ep)."""
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    modality_encoder = build_modality_encoder(
        vision_encoder, language_model, modality="vision"
    )
    plan = ParallelizationPlan(global_ranks=[0, 1])
    plan.parallelize(modality_encoder, ParallelConfig(data_parallel_size=1))  # rpr=1
    plan.parallelize(
        language_model, ParallelConfig(tensor_parallel_size=2, data_parallel_size=1)
    )  # rpr=2
    with pytest.raises(ValueError, match="equal\\s+ranks_per_replica"):
        plan._assign_ranks([0, 1], 2)
