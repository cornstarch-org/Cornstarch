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


def _document_chains(metadata: Any) -> tuple[tuple[int, ...], ...]:
    chains: list[list[int]] = []
    for run in metadata.runs:
        if run.predecessor is None:
            chains.append([])
        chains[-1].append(run.index)
    return tuple(tuple(chain) for chain in chains)


def _merge_prefixes(
    summaries: torch.Tensor,
    chains: tuple[tuple[int, ...], ...],
    *,
    transpose_state_layout: bool,
) -> tuple[torch.Tensor | None, dict[int, int]]:
    """Run FLA's intracard merge kernel and map each non-first run to a row."""
    import triton
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    non_first = sum(max(len(chain) - 1, 0) for chain in chains)
    if non_first == 0:
        return None, {}

    sequence_offsets = [0]
    initial_offsets = [0]
    state_rows: dict[int, int] = {}
    for chain in chains:
        sequence_offsets.append(sequence_offsets[-1] + len(chain))
        base = initial_offsets[-1]
        for position, run_index in enumerate(chain[1:]):
            state_rows[run_index] = base + position
        initial_offsets.append(base + max(len(chain) - 1, 0))

    device = summaries.device
    integer_data = sequence_offsets + initial_offsets + list(range(len(chains)))
    integer_tensor = torch.tensor(integer_data, dtype=torch.int32, device=device)
    sequence_count = len(sequence_offsets)
    initial_count = len(initial_offsets)
    sequence_offsets_tensor = integer_tensor[:sequence_count]
    initial_offsets_tensor = integer_tensor[
        sequence_count : sequence_count + initial_count
    ]
    sequence_ids = integer_tensor[sequence_count + initial_count :]

    heads, key_dim = summaries.shape[1:3]
    value_dim = summaries.shape[3] - key_dim
    state_shape = (
        (non_first, heads, value_dim, key_dim)
        if transpose_state_layout
        else (non_first, heads, key_dim, value_dim)
    )
    states = summaries.new_empty(state_shape, dtype=torch.float32)
    block_key = triton.next_power_of_2(key_dim)

    def grid(meta: dict[str, int]) -> tuple[int, int, int]:
        return (triton.cdiv(value_dim, meta["BV"]), len(chains), heads)

    merge_fwd_bwd_kernel[grid](
        h=states,
        ag_hm=summaries,
        pre_or_post_num_ranks=len(chains),
        rank=0,
        seq_offsets=sequence_offsets_tensor,
        init_offsets=initial_offsets_tensor,
        h0_seq_ids=sequence_ids,
        h0=None,
        HV=heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        FORWARD=True,
        INTRACARD_MODE=True,
        NUM_SEQ_ENTRIES=len(chains),
        TRANSPOSE_STATE=transpose_state_layout,
    )
    return states, state_rows


def _all_reduce_summaries(
    local: torch.Tensor,
    context: CornstarchRunAwareFLACPContext,
) -> torch.Tensor:
    global_summaries = local.new_zeros(
        len(context.metadata.runs), *local.shape[1:]
    )
    if context.local_run_indices:
        indices = torch.tensor(
            context.local_run_indices, dtype=torch.long, device=local.device
        )
        global_summaries.index_copy_(0, indices, local)
    if dist.get_world_size(context.group) > 1:
        dist.all_reduce(global_summaries, group=context.group)
    return global_summaries


def _local_states(
    merged: torch.Tensor | None,
    state_rows: dict[int, int],
    context: CornstarchRunAwareFLACPContext,
    *,
    heads: int,
    key_dim: int,
    value_dim: int,
    transpose_state_layout: bool,
    like: torch.Tensor,
) -> torch.Tensor:
    shape = (
        (len(context.local_run_indices), heads, value_dim, key_dim)
        if transpose_state_layout
        else (len(context.local_run_indices), heads, key_dim, value_dim)
    )
    states = like.new_zeros(shape, dtype=torch.float32)
    if merged is None:
        return states
    destination: list[int] = []
    source: list[int] = []
    for local_index, global_index in enumerate(context.local_run_indices):
        row = state_rows.get(global_index)
        if row is not None:
            destination.append(local_index)
            source.append(row)
    if destination:
        states[torch.tensor(destination, device=like.device)] = merged[
            torch.tensor(source, device=like.device)
        ]
    return states


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
    chains = _document_chains(context.metadata)
    merged, state_rows = _merge_prefixes(
        global_summaries,
        chains,
        transpose_state_layout=transpose_state_layout,
    )
    return _local_states(
        merged,
        state_rows,
        context,
        heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        transpose_state_layout=transpose_state_layout,
        like=k,
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

    # The merge kernel composes prefixes.  Reverse each document chain so those
    # prefixes are precisely the recurrent-gradient suffixes needed by bwd.
    forward_chains = _document_chains(context.metadata)
    reverse_chains = tuple(tuple(reversed(chain)) for chain in forward_chains)
    permutation = [run_index for chain in reverse_chains for run_index in chain]
    reverse_summaries = global_summaries[
        torch.tensor(permutation, dtype=torch.long, device=q.device)
    ]
    contiguous_reverse_chains: list[tuple[int, ...]] = []
    offset = 0
    original_for_contiguous: dict[int, int] = {}
    for chain in reverse_chains:
        contiguous = tuple(range(offset, offset + len(chain)))
        contiguous_reverse_chains.append(contiguous)
        for contiguous_index, original_index in zip(contiguous, chain):
            original_for_contiguous[contiguous_index] = original_index
        offset += len(chain)
    merged, contiguous_rows = _merge_prefixes(
        reverse_summaries,
        tuple(contiguous_reverse_chains),
        transpose_state_layout=transpose_state_layout,
    )
    state_rows = {
        original_for_contiguous[index]: row
        for index, row in contiguous_rows.items()
    }
    local_dht = _local_states(
        merged,
        state_rows,
        context,
        heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        transpose_state_layout=transpose_state_layout,
        like=q,
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
