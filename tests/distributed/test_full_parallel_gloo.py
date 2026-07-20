"""One-GPU Gloo release gates for Qwen's combined parallel topologies."""
from __future__ import annotations

import os
import unittest
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.testing._internal.common_distributed import TIMEOUT_OVERRIDE

from examples.distributed.validate_qwen_full_parallel import (
    CASES,
    _run,
    _validate_environment,
)
from tests.distributed.distributed_base import GlooDistributedTestBase


def _has_fla_gpu() -> bool:
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= 1
            and importlib_metadata.version("flash-linear-attention") == "0.5.0"
        )
    except importlib_metadata.PackageNotFoundError:
        return False


def _args(case_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        case=case_name,
        splitter="head-tail",
        batch_size=2,
        sequence_length=16,
    )


class _OneGpuGlooTestBase(GlooDistributedTestBase):
    """Limit CPU oversubscription while every rank shares ``cuda:0``."""

    def setUp(self) -> None:
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
            super().setUp()

    def _run_case(self, case_name: str) -> None:
        case = CASES[case_name]
        args = _args(case_name)
        _validate_environment(case, args)
        report = _run(case, args)
        self.assertEqual(report["status"], "passed")


# Model import, 5-D process-group construction, and first-use kernel compilation
# can exceed MultiProcessTestCase's generic five-minute timeout on a shared GPU.
TIMEOUT_OVERRIDE.update(
    {
        "test_moe_hybrid_5d": 900,
        "test_moe_attention_only_5d": 900,
        "test_dense_hybrid_4d": 900,
    }
)


@unittest.skipUnless(_has_fla_gpu(), "requires one CUDA GPU and FLA 0.5.0")
class TestQwenMoeFullParallelGloo(_OneGpuGlooTestBase):
    """DP2 x PP2 x CP2 x TP2 x EP2 on 32 Gloo ranks."""

    @property
    def world_size(self) -> int:
        return 32

    def test_moe_hybrid_5d(self) -> None:
        self._run_case("moe-hybrid-5d")

    def test_moe_attention_only_5d(self) -> None:
        self._run_case("moe-attention-5d")


@unittest.skipUnless(_has_fla_gpu(), "requires one CUDA GPU and FLA 0.5.0")
class TestQwenDenseFullParallelGloo(_OneGpuGlooTestBase):
    """DP2 x PP2 x CP2 x TP2 on 16 Gloo ranks."""

    @property
    def world_size(self) -> int:
        return 16

    def test_dense_hybrid_4d(self) -> None:
        self._run_case("dense-hybrid-4d")


if __name__ == "__main__":
    unittest.main()
