"""Structural/reference tests for run-aware Qwen3.5 GDN context metadata."""
from __future__ import annotations

import inspect
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist

from cornstarch.distributed.context_parallel.gated_delta import (
    GatedDeltaNetContextParallelNotSupportedError,
    _compose_initial_states,
    _flatten_local_runs,
    _restore_local_runs,
    _run_aware_convolution,
    _run_fla_fragments,
    _run_summary,
    build_gated_delta_metadata,
    validate_gated_delta_backend,
)
from cornstarch.distributed.context_parallel.gated_delta_fla import (
    CornstarchRunAwareFLACPContext,
    RunAwareFLAContractError,
    _run_aware_backward_preprocess,
    _run_aware_forward_preprocess,
    install_run_aware_fla_dispatch,
)


class _FakeTritonKernel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append({"grid": grid, **kwargs})

        return launch


def _fake_fla_modules(**kernels: object) -> dict[str, ModuleType]:
    fla = ModuleType("fla")
    fla.__path__ = []
    ops = ModuleType("fla.ops")
    ops.__path__ = []
    cp = ModuleType("fla.ops.cp")
    cp.__path__ = []
    chunk = ModuleType("fla.ops.cp.chunk_delta_h")
    for name, kernel in kernels.items():
        setattr(chunk, name, kernel)
    return {
        "fla": fla,
        "fla.ops": ops,
        "fla.ops.cp": cp,
        "fla.ops.cp.chunk_delta_h": chunk,
    }


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


def test_flatten_restore_uses_local_run_order_and_leaves_padding_zero() -> None:
    metadata = build_gated_delta_metadata(
        (torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5])),
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool),
        document_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1]]),
    )
    local = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    flattened = _flatten_local_runs(local, metadata, rank=0)
    assert flattened[:, :, 0].tolist() == [[0.0, 2.0, 4.0]]
    restored = _restore_local_runs(flattened + 10, local, metadata, rank=0)
    assert restored[:, :, 0].tolist() == [[10.0, 12.0, 14.0, 0.0]]


def test_all_padding_lane_uses_differentiable_dummy() -> None:
    metadata = build_gated_delta_metadata(
        (torch.tensor([0, 1]), torch.tensor([2, 3])),
        torch.zeros(1, 4, dtype=torch.bool),
    )
    local = torch.randn(1, 2, 3, requires_grad=True)
    flattened = _flatten_local_runs(local, metadata, rank=0)
    assert flattened.shape == (1, 1, 3)
    assert torch.count_nonzero(flattened) == 0
    restored = _restore_local_runs(flattened, local, metadata, rank=0)
    restored.sum().backward()
    assert local.grad is not None
    assert torch.count_nonzero(local.grad) == 0


def test_globally_all_padding_adapter_keeps_collective_order() -> None:
    metadata = build_gated_delta_metadata(
        (torch.tensor([0, 1]), torch.tensor([2, 3])),
        torch.zeros(1, 4, dtype=torch.bool),
    )
    context = CornstarchRunAwareFLACPContext(
        group=Mock(),
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_cpu=torch.tensor([0, 1], dtype=torch.int32),
        metadata=metadata,
        local_run_indices=(-1,),
    )
    forward_kernel = _FakeTritonKernel()
    backward_kernel = _FakeTritonKernel()
    merge_kernel = _FakeTritonKernel()
    modules = _fake_fla_modules(
        pre_process_fwd_kernel_merged=forward_kernel,
        pre_process_bwd_kernel_merged=backward_kernel,
        merge_fwd_bwd_kernel=merge_kernel,
    )
    collectives: list[tuple[int, ...]] = []
    tensor = torch.zeros(1, 1, 1, 1)
    with (
        patch.dict(sys.modules, modules),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta_fla."
            "dist.get_world_size",
            return_value=2,
        ),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta_fla."
            "dist.all_reduce",
            side_effect=lambda value, group: collectives.append(tuple(value.shape)),
        ),
    ):
        _run_aware_forward_preprocess(
            k=tensor,
            w=tensor,
            u=tensor,
            g=tensor[..., 0],
            cu_seqlens=context.cu_seqlens,
            use_exp2=True,
            initial_state=None,
            context=context,
            transpose_state_layout=False,
        )
        _run_aware_backward_preprocess(
            q=tensor,
            k=tensor,
            w=tensor,
            do=tensor,
            dv=tensor,
            g=tensor[..., 0],
            scale=1.0,
            cu_seqlens=context.cu_seqlens,
            use_exp2=True,
            dht=None,
            context=context,
            transpose_state_layout=False,
        )
    assert collectives == [(1, 1, 1, 2), (1, 1, 1, 2)]
    assert len(forward_kernel.calls) == len(backward_kernel.calls) == 1
    assert merge_kernel.calls == []


def test_fla_adapter_uses_logical_run_prefixes_suffixes_and_boundaries() -> None:
    metadata = build_gated_delta_metadata(
        (torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5])),
        torch.ones(1, 8, dtype=torch.bool),
    )
    context = CornstarchRunAwareFLACPContext(
        group=Mock(),
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_cpu=torch.tensor([0, 2, 4], dtype=torch.int32),
        metadata=metadata,
        local_run_indices=(0, 2),
    )
    forward_kernel = _FakeTritonKernel()
    backward_kernel = _FakeTritonKernel()
    merge_kernel = _FakeTritonKernel()
    modules = _fake_fla_modules(
        pre_process_fwd_kernel_merged=forward_kernel,
        pre_process_bwd_kernel_merged=backward_kernel,
        merge_fwd_bwd_kernel=merge_kernel,
    )
    tensor = torch.zeros(1, 4, 1, 1)
    with (
        patch.dict(sys.modules, modules),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta_fla."
            "dist.get_world_size",
            return_value=1,
        ),
    ):
        forward_states = _run_aware_forward_preprocess(
            k=tensor,
            w=tensor,
            u=tensor,
            g=tensor[..., 0],
            cu_seqlens=context.cu_seqlens,
            use_exp2=True,
            initial_state=None,
            context=context,
            transpose_state_layout=False,
        )
        backward_states, initial_state = _run_aware_backward_preprocess(
            q=tensor,
            k=tensor,
            w=tensor,
            do=tensor,
            dv=tensor,
            g=tensor[..., 0],
            scale=1.0,
            cu_seqlens=context.cu_seqlens,
            use_exp2=True,
            dht=None,
            context=context,
            transpose_state_layout=False,
        )

    assert forward_states.shape == backward_states.shape == (2, 1, 1, 1)
    assert initial_state is None
    assert len(forward_kernel.calls) == 1
    assert forward_kernel.calls[0]["MULTI_SEQS"] is True
    assert torch.equal(forward_kernel.calls[0]["cu_seqlens"], context.cu_seqlens)
    assert [call["cu_seqlens"].tolist() for call in backward_kernel.calls] == [
        [0, 2],
        [2, 4],
    ]
    assert [
        (call["FORWARD"], call["rank"], call["pre_or_post_num_ranks"])
        for call in merge_kernel.calls
    ] == [(True, 2, 2), (False, 0, 2)]
    assert all(call["INTRACARD_MODE"] is False for call in merge_kernel.calls)


def test_fla_dispatch_preserves_stock_context_hooks() -> None:
    fla = ModuleType("fla")
    fla.__path__ = []
    ops = ModuleType("fla.ops")
    ops.__path__ = []
    gated_parent = ModuleType("fla.ops.gated_delta_rule")
    gated_parent.__path__ = []
    gated_chunk = ModuleType("fla.ops.gated_delta_rule.chunk")
    gated_parent.chunk = gated_chunk
    calls: list[str] = []

    def stock_forward(*args, **kwargs):
        calls.append("stock_forward")
        return "stock"

    def stock_backward(*args, **kwargs):
        calls.append("stock_backward")
        return "stock_bwd"

    def stock_identity(value, *, context):
        calls.append("stock_identity")
        return value

    gated_chunk.chunk_gated_delta_rule_fwd_h_pre_process = stock_forward
    gated_chunk.chunk_gated_delta_rule_bwd_dhu_pre_process = stock_backward
    gated_chunk.compress_h0 = stock_identity
    gated_chunk.expand_h0 = stock_identity
    modules = {
        "fla": fla,
        "fla.ops": ops,
        "fla.ops.gated_delta_rule": gated_parent,
        "fla.ops.gated_delta_rule.chunk": gated_chunk,
    }
    with (
        patch.dict(sys.modules, modules),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta_fla."
            "validate_run_aware_fla_contract"
        ),
    ):
        install_run_aware_fla_dispatch()
        ordinary_context = object()
        assert gated_chunk.chunk_gated_delta_rule_fwd_h_pre_process(
            context=ordinary_context
        ) == "stock"
        assert gated_chunk.chunk_gated_delta_rule_bwd_dhu_pre_process(
            context=ordinary_context
        ) == "stock_bwd"
        marker = torch.tensor(1)
        assert gated_chunk.compress_h0(marker, context=ordinary_context) is marker
        assert gated_chunk.expand_h0(marker, context=ordinary_context) is marker
    assert calls == [
        "stock_forward",
        "stock_backward",
        "stock_identity",
        "stock_identity",
    ]


def test_headtail_hybrid_path_never_redistributes_full_activations() -> None:
    """GDN keeps the full-attention head-tail layout across a hybrid boundary."""
    offsets = (torch.tensor([0, 1, 6, 7]), torch.tensor([2, 3, 4, 5]))
    metadata = build_gated_delta_metadata(
        offsets, torch.ones(1, 8, dtype=torch.bool)
    )
    mixed_qkv = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    weight = torch.ones(2, 3)
    forbidden = RuntimeError("full activation redistribution is forbidden")
    with (
        patch(
            "cornstarch.distributed.context_parallel.gated_delta.dist.get_rank",
            return_value=0,
        ),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta."
            "_all_reduce_autograd",
            side_effect=lambda tensor, group: tensor,
        ),
        patch.object(dist, "all_gather", side_effect=forbidden),
        patch.object(dist, "all_gather_into_tensor", side_effect=forbidden),
        patch.object(dist, "all_gather_object", side_effect=forbidden),
        patch.object(dist, "all_to_all", side_effect=forbidden),
        patch.object(dist, "all_to_all_single", side_effect=forbidden),
    ):
        output = _run_aware_convolution(
            mixed_qkv, weight, None, metadata, Mock()
        )
    assert output.shape == mixed_qkv.shape
    production_source = inspect.getsource(_run_aware_convolution) + inspect.getsource(
        _run_fla_fragments
    )
    assert "all_gather" not in production_source
    assert "all_to_all" not in production_source


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
            match="==0.5.0",
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


def test_missing_run_aware_fla_internals_fail_explicitly() -> None:
    def fla_operator(
        q,
        k,
        v,
        *,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cp_context=None,
    ):
        return q, None

    fla_operator.__module__ = "fla.ops.gated_delta_rule.chunk"
    with (
        patch(
            "cornstarch.distributed.context_parallel.gated_delta."
            "importlib_metadata.version",
            return_value="0.5.0",
        ),
        patch(
            "cornstarch.distributed.context_parallel.gated_delta_fla."
            "validate_run_aware_fla_contract",
            side_effect=RunAwareFLAContractError("missing MULTI_SEQS"),
        ),
        pytest.raises(
            GatedDeltaNetContextParallelNotSupportedError,
            match="run-aware forward/backward CP kernels",
        ),
    ):
        validate_gated_delta_backend(
            Mock(chunk_gated_delta_rule=fla_operator)
        )
