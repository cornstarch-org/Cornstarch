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
