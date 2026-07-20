"""Structural/reference tests for run-aware Qwen3.5 GDN context metadata."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch

from cornstarch.distributed.context_parallel.gated_delta import (
    GatedDeltaNetContextParallelNotSupportedError,
    _compose_initial_states,
    _run_summary,
    build_gated_delta_metadata,
    validate_gated_delta_backend,
)


def test_headtail_run_order_predecessors_and_adjacent_merge() -> None:
    offsets = (torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5]))
    metadata = build_gated_delta_metadata(
        offsets,
        torch.ones(1, 8, dtype=torch.bool),
    )
    assert [
        (r.rank, r.local_start, r.length, r.global_start, r.predecessor)
        for r in metadata.runs
    ] == [
        (0, 0, 2, 0, None),
        (1, 0, 4, 2, 0),  # the adjacent central head/tail chunks merge
        (0, 2, 2, 6, 1),
    ]


def test_packed_documents_reset_recurrent_predecessors() -> None:
    offsets = (torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5]))
    document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1]])
    metadata = build_gated_delta_metadata(
        offsets,
        document_ids != -1,
        document_ids=document_ids,
    )
    assert [(r.document_id, r.global_start, r.global_end, r.predecessor) for r in metadata.runs] == [
        (0, 0, 2, None),
        (0, 2, 4, 0),
        (1, 4, 6, None),
        (1, 6, 7, 2),
    ]


def _reference_state(
    key: torch.Tensor,
    value: torch.Tensor,
    decay_log: torch.Tensor,
    beta: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    state = initial
    for token in range(key.shape[0]):
        state = state * decay_log[token].exp()[:, None, None]
        memory = (state * key[token, :, :, None]).sum(-2)
        delta = (value[token] - memory) * beta[token, :, None]
        state = state + key[token, :, :, None] * delta[:, None, :]
    return state


def test_affine_summary_reference_forward_and_backward() -> None:
    """Reference-only proof for the summary algebra used around the FLA op."""
    torch.manual_seed(4)
    key = torch.randn(5, 2, 3, dtype=torch.double, requires_grad=True)
    key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    value = torch.randn(5, 2, 4, dtype=torch.double, requires_grad=True)
    decay = -torch.rand(5, 2, dtype=torch.double, requires_grad=True)
    beta = torch.sigmoid(torch.randn(5, 2, dtype=torch.double, requires_grad=True))
    initial = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    transition, extension = _run_summary(key, value, decay, beta)
    summarized = transition.double() @ initial + extension.double()
    reference = _reference_state(key, value, decay, beta, initial)
    torch.testing.assert_close(summarized, reference, atol=2e-6, rtol=2e-6)
    reference_grad = torch.autograd.grad(reference.sum(), initial, retain_graph=True)[0]
    summary_grad = torch.autograd.grad(summarized.sum(), initial)[0]
    torch.testing.assert_close(summary_grad, reference_grad, atol=2e-6, rtol=2e-6)


def test_prefix_composition_resets_documents() -> None:
    metadata = build_gated_delta_metadata(
        (torch.tensor([0, 2]), torch.tensor([1, 3])),
        torch.ones(1, 4, dtype=torch.bool),
        document_ids=torch.tensor([[0, 0, 1, 1]]),
    )
    transitions = torch.full((4, 1, 1, 1), 2.0)
    extensions = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1, 1, 1)
    states = _compose_initial_states(transitions, extensions, metadata)
    assert states[:, 0, 0, 0].tolist() == [0.0, 1.0, 0.0, 3.0]


def test_missing_fla_fails_explicitly() -> None:
    with patch(
        "cornstarch.distributed.context_parallel.gated_delta.importlib_metadata.version",
        side_effect=__import__("importlib.metadata").metadata.PackageNotFoundError,
    ):
        with pytest.raises(
            GatedDeltaNetContextParallelNotSupportedError,
            match="requires flash-linear-attention",
        ):
            validate_gated_delta_backend(Mock())


def test_unsupported_fla_version_fails_explicitly() -> None:
    operator = Mock()
    operator.__module__ = "fla.ops.gated_delta_rule"
    with patch(
        "cornstarch.distributed.context_parallel.gated_delta.importlib_metadata.version",
        return_value="0.4.2",
    ):
        with pytest.raises(
            GatedDeltaNetContextParallelNotSupportedError,
            match=">=0.5.0,<0.6",
        ):
            validate_gated_delta_backend(Mock(chunk_gated_delta_rule=operator))


def test_transformers_torch_fallback_fails_explicitly() -> None:
    def torch_fallback(
        q, k, v, *, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=False
    ):
        return q, None

    with patch(
        "cornstarch.distributed.context_parallel.gated_delta.importlib_metadata.version",
        return_value="0.5.0",
    ):
        with pytest.raises(
            GatedDeltaNetContextParallelNotSupportedError,
            match="operator module",
        ):
            validate_gated_delta_backend(
                Mock(chunk_gated_delta_rule=torch_fallback)
            )
