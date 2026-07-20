"""Run-aware context parallelism for Qwen3.5 Gated DeltaNet layers.

The stock ``FLACPContext`` derives one contiguous interval from the physical
rank, which is not valid for Cornstarch's head-tail token ownership.  This
module keeps ownership metadata outside the reused Hugging Face model and
delegates recurrent summary generation/composition to FLA's optimized CP
kernels through Cornstarch's run-aware FLA adapter.

No token activation is gathered.  Recurrent collectives carry FLA's compact
``(M, S)`` summaries; convolution exchanges at most ``kernel_size - 1``
projected tokens per run.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from types import MethodType
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from packaging.version import InvalidVersion, Version

from cornstarch.distributed.context_parallel.splitters import (
    HeadTailContextParallelSplitter,
    UniformContextParallelSplitter,
)


class GatedDeltaNetContextParallelNotSupportedError(NotImplementedError):
    """Raised when exact distributed Gated DeltaNet execution is unavailable."""


@dataclass(frozen=True)
class GatedDeltaRun:
    """One maximal, document-local contiguous run owned by a CP rank."""

    index: int
    rank: int
    batch_index: int
    document_id: int
    local_start: int
    length: int
    global_start: int
    global_end: int
    predecessor: int | None
    successor: int | None


@dataclass(frozen=True)
class GatedDeltaCPMetadata:
    """Immutable per-microbatch ownership and packed-document metadata."""

    offsets_per_rank: tuple[torch.Tensor, ...]
    document_ids: torch.Tensor
    runs: tuple[GatedDeltaRun, ...]

    def local_runs(self, rank: int) -> tuple[GatedDeltaRun, ...]:
        return tuple(run for run in self.runs if run.rank == rank)


@dataclass
class _RunDraft:
    """Mutable run used only while predecessor links are being assembled."""

    rank: int
    batch_index: int
    document_id: int
    local_start: int
    length: int
    global_start: int
    global_end: int
    predecessor: int | None = None
    successor: int | None = None


def build_gated_delta_metadata(
    offsets_per_rank: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    *,
    document_ids: torch.Tensor | None = None,
) -> GatedDeltaCPMetadata:
    """Build globally ordered maximal runs from a splitter's actual offsets.

    ``document_ids`` uses ``-1`` for padding.  When absent, each batch item is
    one document and the 2-D attention mask identifies padding.  Explicit IDs
    allow packed samples to reset both the recurrence and causal convolution.
    """
    batch, seq_len = _validate_gated_delta_inputs(attention_mask, document_ids)
    document_ids = _normalize_document_ids(attention_mask, document_ids)
    offsets = _normalize_offsets(offsets_per_rank, seq_len)
    drafts = _find_gated_delta_runs(offsets, document_ids, batch)
    _link_gated_delta_runs(drafts)
    return GatedDeltaCPMetadata(
        offsets_per_rank=offsets,
        document_ids=document_ids,
        runs=tuple(
            GatedDeltaRun(index=index, **vars(draft))
            for index, draft in enumerate(drafts)
        ),
    )


def _validate_gated_delta_inputs(
    attention_mask: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> tuple[int, int]:
    if attention_mask.ndim != 2:
        raise ValueError(
            "Gated DeltaNet CP requires a 2-D padding/document mask; "
            f"got shape {tuple(attention_mask.shape)}."
        )
    batch, seq_len = attention_mask.shape
    if document_ids is not None and document_ids.shape != (batch, seq_len):
        raise ValueError(
            "document_ids must match the unsplit attention mask shape; "
            f"got {tuple(document_ids.shape)} and {(batch, seq_len)}."
        )
    return batch, seq_len


def _normalize_document_ids(
    attention_mask: torch.Tensor,
    document_ids: torch.Tensor | None,
) -> torch.Tensor:
    if document_ids is None:
        document_ids = torch.where(
            attention_mask.to(dtype=torch.bool),
            torch.zeros_like(attention_mask, dtype=torch.long),
            torch.full_like(attention_mask, -1, dtype=torch.long),
        )
    return document_ids.detach().to(device="cpu", dtype=torch.long)


def _normalize_offsets(
    offsets_per_rank: Sequence[torch.Tensor], seq_len: int
) -> tuple[torch.Tensor, ...]:
    offsets = tuple(
        value.detach().to(device="cpu", dtype=torch.long).contiguous()
        for value in offsets_per_rank
    )
    if offsets and sorted(torch.cat(offsets).tolist()) != list(range(seq_len)):
        raise ValueError(
            "CP splitter offsets must assign every global token exactly once."
        )
    return offsets


def _find_gated_delta_runs(
    offsets: tuple[torch.Tensor, ...],
    document_ids: torch.Tensor,
    batch: int,
) -> list[_RunDraft]:
    """Scan each rank's ordered offsets into maximal document-local runs."""
    drafts: list[_RunDraft] = []
    for batch_index in range(batch):
        for rank, rank_offsets in enumerate(offsets):
            values = rank_offsets.tolist()
            local_index = 0
            while local_index < len(values):
                global_start = values[local_index]
                document_id = int(document_ids[batch_index, global_start])
                if document_id < 0:
                    local_index += 1
                    continue
                end = local_index + 1
                while end < len(values):
                    position = values[end]
                    previous = values[end - 1]
                    if (
                        position != previous + 1
                        or int(document_ids[batch_index, position]) != document_id
                    ):
                        break
                    end += 1
                drafts.append(
                    _RunDraft(
                        rank=rank,
                        batch_index=batch_index,
                        document_id=document_id,
                        local_start=local_index,
                        length=end - local_index,
                        global_start=global_start,
                        global_end=values[end - 1] + 1,
                    )
                )
                local_index = end
    return drafts


def _link_gated_delta_runs(drafts: list[_RunDraft]) -> None:
    """Order drafts globally and connect consecutive runs of one document."""
    drafts.sort(
        key=lambda run: (run.batch_index, run.document_id, run.global_start)
    )
    for index, (previous, run) in enumerate(zip(drafts, drafts[1:]), start=1):
        if (
            previous.batch_index == run.batch_index
            and previous.document_id == run.document_id
        ):
            run.predecessor = index - 1
            previous.successor = index


def validate_gated_delta_backend(linear_attn: torch.nn.Module) -> None:
    """Require the pinned FLA operator contract used by the run adapter."""
    distribution_name = None
    version = None
    for candidate in ("flash-linear-attention", "fla-core"):
        try:
            version = importlib_metadata.version(candidate)
            distribution_name = candidate
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    if version is None:
        raise GatedDeltaNetContextParallelNotSupportedError(
            "Qwen3.5 linear-attention context parallelism requires "
            "flash-linear-attention==0.5.0. Install Cornstarch with the "
            "FLA CUDA backend; independent local recurrent states are not a "
            "correct fallback."
        )
    try:
        parsed_version = Version(version)
    except InvalidVersion as exc:
        raise GatedDeltaNetContextParallelNotSupportedError(
            f"Cannot validate {distribution_name} version {version!r}."
        ) from exc
    if parsed_version != Version("0.5.0"):
        raise GatedDeltaNetContextParallelNotSupportedError(
            "Qwen3.5 linear-attention CP supports only the audited "
            f"flash-linear-attention==0.5.0 contract; detected "
            f"{distribution_name}=={version}."
        )

    operator = getattr(linear_attn, "chunk_gated_delta_rule", None)
    module_name = getattr(operator, "__module__", "")
    try:
        parameters = inspect.signature(operator).parameters
    except (TypeError, ValueError):
        parameters = {}
    required = {
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "cp_context",
    }
    if not module_name.startswith("fla.") or not required.issubset(parameters):
        raise GatedDeltaNetContextParallelNotSupportedError(
            "The installed flash-linear-attention backend does not expose the "
            "differentiable Gated DeltaNet initial-state contract required for "
            f"run-aware CP (detected version {version!r}, operator module "
            f"{module_name!r})."
        )
    try:
        from cornstarch.distributed.context_parallel.gated_delta_fla import (
            RunAwareFLAContractError,
            validate_run_aware_fla_contract,
        )

        validate_run_aware_fla_contract()
    except RunAwareFLAContractError as exc:
        raise GatedDeltaNetContextParallelNotSupportedError(
            "The installed flash-linear-attention backend lacks Cornstarch's "
            f"required run-aware forward/backward CP kernels: {exc}"
        ) from exc


def _all_reduce_autograd(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    if dist.get_world_size(group) == 1:
        return tensor
    return dist_nn.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)


def _run_summary(
    key: torch.Tensor,
    value: torch.Tensor,
    decay_log: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference-only affine recurrence used by CPU algebra tests.

    Production CP must use :mod:`gated_delta_fla`; this intentionally slow
    token loop is kept only to prove the summary algebra without CUDA/FLA.
    """
    key = key.float()
    value = value.float()
    decay_log = decay_log.float()
    beta = beta.float()
    heads, key_dim, value_dim = key.shape[1], key.shape[2], value.shape[2]
    transition = torch.eye(key_dim, device=key.device, dtype=torch.float32)
    transition = transition.unsqueeze(0).expand(heads, -1, -1).clone()
    extension = torch.zeros(
        heads, key_dim, value_dim, device=key.device, dtype=torch.float32
    )
    identity = torch.eye(key_dim, device=key.device, dtype=torch.float32).unsqueeze(0)
    for token in range(key.shape[0]):
        k_t = key[token]
        beta_t = beta[token, :, None, None]
        decay_t = decay_log[token].exp()[:, None, None]
        token_transition = decay_t * (
            identity - beta_t * k_t[:, :, None] * k_t[:, None, :]
        )
        token_extension = beta_t * k_t[:, :, None] * value[token, :, None, :]
        extension = token_transition @ extension + token_extension
        transition = token_transition @ transition
    return transition, extension


def _compose_initial_states(
    transitions: torch.Tensor,
    extensions: torch.Tensor,
    metadata: GatedDeltaCPMetadata,
) -> torch.Tensor:
    """Compose globally ordered prefixes, resetting at document boundaries."""
    states = torch.zeros_like(extensions)
    for run in metadata.runs:
        if run.predecessor is None:
            continue
        previous = run.predecessor
        states[run.index] = (
            transitions[previous] @ states[previous] + extensions[previous]
        )
    return states


def _run_aware_convolution(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    metadata: GatedDeltaCPMetadata,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Apply causal depthwise convolution with one communicated halo per run."""
    batch, channels, local_seq = mixed_qkv.shape
    kernel_size = weight.shape[-1]
    halo_size = kernel_size - 1
    rank = dist.get_rank(cp_group)
    if halo_size == 0:
        return F.silu(F.conv1d(mixed_qkv, weight[:, None], bias, groups=channels))

    tails = mixed_qkv.new_zeros(
        (len(metadata.runs), halo_size, channels)
    )
    for run in metadata.local_runs(rank):
        values = mixed_qkv[
            run.batch_index,
            :,
            run.local_start : run.local_start + run.length,
        ].transpose(0, 1)
        take = min(halo_size, run.length)
        tails[run.index, -take:] = values[-take:]
    tails = _all_reduce_autograd(tails, cp_group)

    # Keep empty/all-padding lanes connected to autograd so they enter the
    # synchronized dummy FLA backward and execute collectives in rank order.
    output = mixed_qkv * 0
    for run in metadata.local_runs(rank):
        prefix_parts: list[torch.Tensor] = []
        remaining = halo_size
        predecessor = run.predecessor
        while predecessor is not None and remaining > 0:
            previous_run = metadata.runs[predecessor]
            take = min(remaining, previous_run.length)
            prefix_parts.append(tails[predecessor, -take:])
            remaining -= take
            predecessor = previous_run.predecessor
        prefix_parts.reverse()
        if remaining:
            prefix_parts.insert(
                0, mixed_qkv.new_zeros((remaining, channels))
            )
        local = mixed_qkv[
            run.batch_index,
            :,
            run.local_start : run.local_start + run.length,
        ].transpose(0, 1)
        convolution_input = torch.cat([*prefix_parts, local], dim=0).transpose(0, 1)
        convolved = F.conv1d(
            convolution_input[None], weight[:, None], bias, groups=channels
        )
        output[
            run.batch_index,
            :,
            run.local_start : run.local_start + run.length,
        ] = F.silu(convolved[0])
    return output


def _flatten_local_runs(
    tensor: torch.Tensor,
    metadata: GatedDeltaCPMetadata,
    rank: int,
) -> torch.Tensor:
    fragments = [
        tensor[
            run.batch_index,
            run.local_start : run.local_start + run.length,
        ]
        for run in metadata.local_runs(rank)
    ]
    if not fragments:
        # FLA varlen requires at least one sequence. A zero dummy remains in the
        # graph but is excluded from the global logical summary table.
        return tensor.sum(dim=(0, 1), keepdim=True) * 0
    return torch.cat(fragments, dim=0).unsqueeze(0).contiguous()


def _restore_local_runs(
    flattened: torch.Tensor,
    like: torch.Tensor,
    metadata: GatedDeltaCPMetadata,
    rank: int,
) -> torch.Tensor:
    # The zero dependency is essential on an all-padding lane: backward must
    # still traverse FLA and participate in summary collectives.
    output = torch.zeros_like(like) + flattened.sum() * 0
    offset = 0
    for run in metadata.local_runs(rank):
        output[
            run.batch_index,
            run.local_start : run.local_start + run.length,
        ] = flattened[0, offset : offset + run.length]
        offset += run.length
    return output


def _run_fla_fragments(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay_log: torch.Tensor,
    beta: torch.Tensor,
    metadata: GatedDeltaCPMetadata,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Run all local fragments in one FLA varlen call with run-aware CP hooks."""
    from cornstarch.distributed.context_parallel.gated_delta_fla import (
        CornstarchRunAwareFLACPContext,
        install_run_aware_fla_dispatch,
    )

    rank = dist.get_rank(cp_group)
    local_runs = metadata.local_runs(rank)
    lengths = [run.length for run in local_runs] or [1]
    cu_seqlens_cpu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    cu_seqlens = cu_seqlens_cpu.to(device=query.device, non_blocking=True)
    context = CornstarchRunAwareFLACPContext(
        group=cp_group,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        metadata=metadata,
        local_run_indices=(
            tuple(run.index for run in local_runs) if local_runs else (-1,)
        ),
    )
    install_run_aware_fla_dispatch()
    run_output, _ = operator(
        _flatten_local_runs(query, metadata, rank),
        _flatten_local_runs(key, metadata, rank),
        _flatten_local_runs(value, metadata, rank),
        g=_flatten_local_runs(decay_log, metadata, rank),
        beta=_flatten_local_runs(beta, metadata, rank),
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        cp_context=context,
    )
    return _restore_local_runs(run_output, value, metadata, rank)


def _distributed_gated_delta_forward(
    linear_attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    metadata: GatedDeltaCPMetadata,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Qwen3.5 GDN forward with run-aware convolution and recurrent state."""
    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask[:, :, None].to(hidden_states.dtype)
    batch_size, seq_len, _ = hidden_states.shape
    mixed_qkv = linear_attn.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv = _run_aware_convolution(
        mixed_qkv,
        linear_attn.conv1d.weight.squeeze(1),
        linear_attn.conv1d.bias,
        metadata,
        cp_group,
    ).transpose(1, 2)

    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn.key_dim, linear_attn.key_dim, linear_attn.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, linear_attn.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn.head_v_dim)
    z = linear_attn.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, linear_attn.head_v_dim
    )
    beta = linear_attn.in_proj_b(hidden_states).sigmoid()
    a = linear_attn.in_proj_a(hidden_states)
    decay_log = -linear_attn.A_log.float().exp() * F.softplus(
        a.float() + linear_attn.dt_bias
    )
    if linear_attn.num_v_heads // linear_attn.num_k_heads > 1:
        repeat = linear_attn.num_v_heads // linear_attn.num_k_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    core = _run_fla_fragments(
        linear_attn.chunk_gated_delta_rule,
        query,
        key,
        value,
        decay_log,
        beta,
        metadata,
        cp_group,
    )
    core = linear_attn.norm(
        core.reshape(-1, linear_attn.head_v_dim),
        z.reshape(-1, linear_attn.head_v_dim),
    ).reshape(batch_size, seq_len, -1)
    return linear_attn.out_proj(core)


def inject_gated_delta_context_parallel(
    module: torch.nn.Module,
    cp_group: dist.ProcessGroup,
    splitter: object,
) -> int:
    """Patch every Qwen3.5 GDN leaf and return the number patched."""
    if not isinstance(
        splitter, (UniformContextParallelSplitter, HeadTailContextParallelSplitter)
    ):
        raise GatedDeltaNetContextParallelNotSupportedError(
            "Qwen3.5 Gated DeltaNet CP supports only uniform and head-tail "
            f"ownership; got {type(splitter).__name__}."
        )
    patched = 0
    for layer in module.modules():
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        validate_gated_delta_backend(linear_attn)
        base_layer_forward = layer.forward

        def layer_forward(
            this: torch.nn.Module,
            *args: Any,
            _base: Callable[..., Any] = base_layer_forward,
            cp_sequence_metadata: GatedDeltaCPMetadata | None = None,
            **kwargs: Any,
        ) -> Any:
            if cp_sequence_metadata is None:
                raise GatedDeltaNetContextParallelNotSupportedError(
                    "Qwen3.5 linear-attention CP did not receive per-microbatch "
                    "run metadata. Use ParallelContext.prepare_dataloader()."
                )
            linear = this.linear_attn
            original = linear.forward

            def distributed_forward(
                _linear: torch.nn.Module,
                hidden_states: torch.Tensor,
                cache_params: Any = None,
                attention_mask: torch.Tensor | None = None,
            ) -> torch.Tensor:
                if cache_params is not None:
                    raise GatedDeltaNetContextParallelNotSupportedError(
                        "Cached decoding is not supported by training-time GDN CP."
                    )
                return _distributed_gated_delta_forward(
                    _linear,
                    hidden_states,
                    attention_mask,
                    cp_sequence_metadata,
                    cp_group,
                )

            linear.forward = MethodType(distributed_forward, linear)
            try:
                return _base(*args, **kwargs)
            finally:
                linear.forward = original

        layer.forward = MethodType(layer_forward, layer)
        patched += 1
    return patched
