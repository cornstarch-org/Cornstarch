"""FLA 0.5 run-aware context-parallel adapter for Gated DeltaNet.

This module deliberately contains no alternate recurrent implementation.  It
installs a narrow dispatch layer around FLA's own GDN custom autograd function:
FLA continues to compute the WY representation and local forward/backward,
while its optimized CP pre-scan and merge kernels communicate state summaries
in the logical run order supplied by Cornstarch.

Imports are lazy so CPU-only users can import Cornstarch without installing
FLA or Triton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


class RunAwareFLAContractError(RuntimeError):
    """Raised when the installed FLA internals do not match the pinned API."""


@dataclass
class CornstarchRunAwareFLACPContext:
    """Metadata passed through FLA's existing GDN custom autograd context."""

    group: dist.ProcessGroup
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    metadata: Any
    # ``-1`` is a synchronized zero dummy for an empty/all-padding CP lane.
    local_run_indices: tuple[int, ...]

    @property
    def num_seqs(self) -> int:
        return len(self.local_run_indices)

    @property
    def is_cp_enabled(self) -> bool:
        return True


_REQUIRED_KERNEL_ARGUMENTS = {
    "pre_process_fwd_kernel_merged": {
        "MULTI_SEQS",
        "USE_EXP2",
        "cu_seqlens",
        "hm",
    },
    "merge_fwd_bwd_kernel": {
        "INTRACARD_MODE",
        "TRANSPOSE_STATE",
        "seq_offsets",
        "init_offsets",
    },
    "pre_process_bwd_kernel_merged": {
        "USE_EXP2",
        "cu_seqlens",
        "dhm",
    },
}


def _kernel_arguments(kernel: Any) -> set[str]:
    arguments = getattr(kernel, "arg_names", None)
    return set(arguments or ())


def validate_run_aware_fla_contract() -> None:
    """Validate the exact optimized kernel surface used by this adapter."""
    try:
        from fla.ops.cp import chunk_delta_h
        from fla.ops.gated_delta_rule import chunk as gated_delta_chunk
    except (ImportError, OSError) as exc:
        raise RunAwareFLAContractError(
            "FLA's Gated DeltaNet CP modules could not be imported."
        ) from exc

    for name, required in _REQUIRED_KERNEL_ARGUMENTS.items():
        kernel = getattr(chunk_delta_h, name, None)
        missing = required - _kernel_arguments(kernel)
        if kernel is None or missing:
            raise RunAwareFLAContractError(
                f"FLA kernel {name!r} is missing the run-aware contract: "
                f"{sorted(missing)}."
            )
    for name in (
        "chunk_gated_delta_rule_fwd_h_pre_process",
        "chunk_gated_delta_rule_bwd_dhu_pre_process",
        "compress_h0",
        "expand_h0",
    ):
        if not callable(getattr(gated_delta_chunk, name, None)):
            raise RunAwareFLAContractError(
                f"FLA GDN autograd hook {name!r} is unavailable."
            )


def _merge_logical_states(
    summaries: torch.Tensor,
    context: CornstarchRunAwareFLACPContext,
    *,
    forward: bool,
    transpose_state_layout: bool,
) -> torch.Tensor:
    """Merge predecessor/successor summaries using runs as logical ranks."""
    import triton
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    heads, key_dim = summaries.shape[1:3]
    value_dim = summaries.shape[3] - key_dim
    state_shape = (
        (len(context.local_run_indices), heads, value_dim, key_dim)
        if transpose_state_layout
        else (len(context.local_run_indices), heads, key_dim, value_dim)
    )
    states = summaries.new_zeros(state_shape, dtype=torch.float32)
    block_key = triton.next_power_of_2(key_dim)

    def grid(meta: dict[str, int]) -> tuple[int, int]:
        return (triton.cdiv(value_dim, meta["BV"]), heads)

    for local_index, global_index in enumerate(context.local_run_indices):
        if global_index < 0:
            continue
        run = context.metadata.runs[global_index]
        linked_count = 0
        linked = run.predecessor if forward else run.successor
        while linked is not None:
            linked_count += 1
            linked_run = context.metadata.runs[linked]
            linked = linked_run.predecessor if forward else linked_run.successor
        if linked_count == 0:
            continue
        merge_fwd_bwd_kernel[grid](
            h=states[local_index],
            ag_hm=summaries,
            pre_or_post_num_ranks=linked_count,
            rank=global_index,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=heads,
            K=key_dim,
            V=value_dim,
            BK=block_key,
            FORWARD=forward,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            TRANSPOSE_STATE=transpose_state_layout,
        )
    return states


def _all_reduce_summaries(
    local: torch.Tensor,
    context: CornstarchRunAwareFLACPContext,
) -> torch.Tensor:
    global_summaries = local.new_zeros(
        max(1, len(context.metadata.runs)), *local.shape[1:]
    )
    valid_local = [
        (local_index, global_index)
        for local_index, global_index in enumerate(context.local_run_indices)
        if global_index >= 0
    ]
    if valid_local:
        indices = torch.tensor(
            [global_index for _, global_index in valid_local],
            dtype=torch.long,
            device=local.device,
        )
        local_indices = torch.tensor(
            [local_index for local_index, _ in valid_local],
            dtype=torch.long,
            device=local.device,
        )
        global_summaries.index_copy_(0, indices, local[local_indices])
    if dist.get_world_size(context.group) > 1:
        dist.all_reduce(global_summaries, group=context.group)
    return global_summaries


def _run_aware_forward_preprocess(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor,
    use_exp2: bool,
    initial_state: torch.Tensor | None,
    context: CornstarchRunAwareFLACPContext,
    transpose_state_layout: bool,
) -> torch.Tensor:
    import triton
    from fla.ops.cp.chunk_delta_h import pre_process_fwd_kernel_merged

    if initial_state is not None:
        raise AssertionError("Run-aware FLA CP owns the recurrent initial states.")
    _, _, key_heads, key_dim = k.shape
    value_heads, value_dim = u.shape[2:]
    if key_dim > 256:
        raise RunAwareFLAContractError(
            "FLA's run-aware pre-scan supports key head dimensions up to 256."
        )
    local_count = len(context.local_run_indices)
    summaries = k.new_empty(
        local_count, value_heads, key_dim, value_dim + key_dim, dtype=torch.float32
    )
    block_size = 32 if key_dim <= 64 else 64
    block_key = triton.next_power_of_2(key_dim)
    grid = (
        triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size),
        value_heads,
        local_count,
    )
    pre_process_fwd_kernel_merged[grid](
        k=k,
        v=u if v is None else v,
        w=w,
        g=g,
        gk=gk,
        bg=bg,
        u=u,
        hm=summaries,
        cu_seqlens=cu_seqlens,
        T=0,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        BK1=block_key,
        BLOCK_SIZE=block_size,
        USE_EXP2=use_exp2,
        MULTI_SEQS=True,
    )
    global_summaries = _all_reduce_summaries(summaries, context)
    return _merge_logical_states(
        global_summaries,
        context,
        forward=True,
        transpose_state_layout=transpose_state_layout,
    )


def _run_aware_backward_preprocess(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float,
    cu_seqlens: torch.Tensor,
    use_exp2: bool,
    dht: torch.Tensor | None,
    context: CornstarchRunAwareFLACPContext,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor, None]:
    import triton
    from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged

    if dht is not None:
        raise AssertionError("Run-aware FLA CP owns the recurrent final gradients.")
    _, _, key_heads, key_dim = q.shape
    value_heads, value_dim = do.shape[2:]
    local_count = len(context.local_run_indices)
    summaries = q.new_empty(
        local_count, value_heads, key_dim, value_dim + key_dim, dtype=torch.float32
    )
    block_size = 32 if key_dim <= 64 else 64
    block_key = triton.next_power_of_2(key_dim)
    grid = (
        triton.cdiv(value_dim, block_size) + triton.cdiv(key_dim, block_size),
        value_heads,
    )
    # FLA 0.5's backward pre-scan is a single-sequence kernel.  Launch it on
    # each local run while retaining one packed local core/autograd invocation.
    for local_index in range(local_count):
        boundaries = cu_seqlens[local_index : local_index + 2]
        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k if bg is None else bg,
            w=w,
            g=g,
            gk=gk,
            do=do,
            dhm=summaries[local_index],
            dv=dv,
            cu_seqlens=boundaries,
            scale=scale,
            T=0,
            H=key_heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=64,
            BK1=block_key,
            BLOCK_SIZE=block_size,
            USE_EXP2=use_exp2,
        )
    global_summaries = _all_reduce_summaries(summaries, context)

    local_dht = _merge_logical_states(
        global_summaries,
        context,
        forward=False,
        transpose_state_layout=transpose_state_layout,
    )
    return local_dht, None


def install_run_aware_fla_dispatch() -> None:
    """Install idempotent type-dispatch hooks into FLA's GDN autograd module."""
    validate_run_aware_fla_contract()
    from fla.ops.gated_delta_rule import chunk as gated_delta_chunk

    if getattr(gated_delta_chunk, "_cornstarch_run_aware_dispatch", False):
        return

    original_forward = gated_delta_chunk.chunk_gated_delta_rule_fwd_h_pre_process
    original_backward = gated_delta_chunk.chunk_gated_delta_rule_bwd_dhu_pre_process
    original_compress = gated_delta_chunk.compress_h0
    original_expand = gated_delta_chunk.expand_h0

    def forward_dispatch(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        if not isinstance(context, CornstarchRunAwareFLACPContext):
            return original_forward(*args, **kwargs)
        return _run_aware_forward_preprocess(**kwargs)

    def backward_dispatch(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        if not isinstance(context, CornstarchRunAwareFLACPContext):
            return original_backward(*args, **kwargs)
        kwargs.pop("initial_state", None)
        return _run_aware_backward_preprocess(**kwargs)

    def compress_dispatch(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        if isinstance(context, CornstarchRunAwareFLACPContext):
            return args[0] if args else kwargs["h0"]
        return original_compress(*args, **kwargs)

    def expand_dispatch(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        if isinstance(context, CornstarchRunAwareFLACPContext):
            return args[0] if args else kwargs["h0"]
        return original_expand(*args, **kwargs)

    gated_delta_chunk.chunk_gated_delta_rule_fwd_h_pre_process = forward_dispatch
    gated_delta_chunk.chunk_gated_delta_rule_bwd_dhu_pre_process = backward_dispatch
    gated_delta_chunk.compress_h0 = compress_dispatch
    gated_delta_chunk.expand_h0 = expand_dispatch
    gated_delta_chunk._cornstarch_run_aware_dispatch = True
