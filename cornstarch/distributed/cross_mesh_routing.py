"""Differentiable N-to-M routing across multimodal context-parallel meshes.

Projected modality rows are identified by their ordinal within each sample and
matched to that sample's ordered global placeholder positions. The source
splitter owns modality-row ordinals while the destination splitter owns merged
text positions; no uniform-chunk or physical-rank arithmetic is used. A route
packs rows by destination, performs one variable-split ``all_to_all_single``,
then restores the destination splitter's local placeholder order.

Pipeline schedules have an explicit autograd graph break, so they call
``exchange_forward`` / ``exchange_backward`` and ship the exact transposed
exchange themselves.  Co-located/direct callers can use ``route_autograd``;
its backward is the same transposed all-to-all and never gathers full features.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


CP_ROUTING_OFFSETS_KEY = "_cornstarch_cp_routing_offsets"
CP_MODALITY_MASKS_KEY = "cp_modality_attention_masks"


@dataclass(frozen=True)
class CrossMeshGroup:
    """One DP-replica and compatible TP/EP-lane seam process group."""

    ranks: tuple[int, ...]
    producer_ranks: tuple[int, ...]
    consumer_ranks: tuple[int, ...]
    process_group: dist.ProcessGroup
    dp_rank: int
    producer_tp_rank: int
    producer_ep_rank: int
    consumer_tp_rank: int
    consumer_ep_rank: int

    @property
    def rank_to_index(self) -> dict[int, int]:
        return {rank: index for index, rank in enumerate(self.ranks)}


@dataclass(frozen=True)
class RoutePlan:
    """Rank-local packing/restoration plan for one :class:`CrossMeshGroup`."""

    source_select_indices: tuple[int, ...]
    send_indices: tuple[int, ...]
    restore_indices: tuple[int, ...]
    send_counts: tuple[int, ...]
    recv_counts: tuple[int, ...]
    source_rows: int
    source_storage_rows: int
    destination_rows: int


@dataclass
class GroupExchangeState:
    """Saved forward layout needed for the exact reverse exchange."""

    seam_group: CrossMeshGroup
    plan: RoutePlan


@dataclass
class SeamExchangeState:
    """All lane exchanges performed by one physical rank for one microbatch."""

    groups: list[GroupExchangeState]
    source_shape: torch.Size | None
    source_fanout: int


class _VariableSplitAllToAll(torch.autograd.Function):
    """Autograd-aware variable row exchange, including zero-sized splits."""

    @staticmethod
    def forward(ctx, input_, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = tuple(output_split_sizes)
        ctx.input_split_sizes = tuple(input_split_sizes)
        ctx.group = group
        output = input_.new_empty((sum(output_split_sizes), *input_.shape[1:]))
        dist.all_to_all_single(
            output,
            input_.contiguous(),
            output_split_sizes=list(output_split_sizes),
            input_split_sizes=list(input_split_sizes),
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_empty(
            (sum(ctx.input_split_sizes), *grad_output.shape[1:])
        )
        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            output_split_sizes=list(ctx.input_split_sizes),
            input_split_sizes=list(ctx.output_split_sizes),
            group=ctx.group,
        )
        return grad_input, None, None, None


def _offset_lists(offsets: Sequence[torch.Tensor]) -> list[list[int]]:
    return [offset.to(device="cpu", dtype=torch.long).tolist() for offset in offsets]


def _placeholder_keys(
    global_input_ids: torch.Tensor,
    token_id: int,
    offsets: Sequence[int],
) -> list[tuple[int, int]]:
    keys: list[tuple[int, int]] = []
    ids = global_input_ids.detach().to(device="cpu")
    for batch_index in range(ids.shape[0]):
        for position in offsets:
            if int(ids[batch_index, position]) == token_id:
                keys.append((batch_index, position))
    return keys


def _placeholder_keys_by_source_ordinal(
    global_input_ids: torch.Tensor,
    token_id: int,
    source_attention_mask: torch.Tensor,
    source_ordinals: Sequence[int],
) -> list[tuple[int, int]]:
    """Map locally owned modality-row ordinals to global placeholder keys."""
    ids = global_input_ids.detach().to(device="cpu")
    keys: list[tuple[int, int]] = []
    for batch_index in range(ids.shape[0]):
        placeholders = (ids[batch_index] == token_id).nonzero(as_tuple=True)[0].tolist()
        valid_ordinals = source_attention_mask[batch_index].nonzero(
            as_tuple=True
        )[0].tolist()
        ordinal_to_placeholder = dict(zip(valid_ordinals, placeholders))
        keys.extend(
            (batch_index, ordinal_to_placeholder[ordinal])
            for ordinal in source_ordinals
            if ordinal in ordinal_to_placeholder
        )
    return keys


def build_route_plan(
    *,
    global_input_ids: torch.Tensor,
    token_id: int,
    source_attention_mask: torch.Tensor | None,
    source_offsets: Sequence[torch.Tensor],
    destination_offsets: Sequence[torch.Tensor],
    seam_group: CrossMeshGroup,
    global_rank: int,
) -> RoutePlan:
    """Derive a deterministic one-row-per-placeholder route from real offsets."""
    if global_input_ids.ndim != 2:
        raise ValueError("cp_global_input_ids must have shape (batch, sequence).")

    source = _offset_lists(source_offsets)
    destination = _offset_lists(destination_offsets)
    seq_len = global_input_ids.shape[1]
    source_flat = [ordinal for rank_offsets in source for ordinal in rank_offsets]
    source_length = max(source_flat, default=-1) + 1
    if sorted(source_flat) != list(range(source_length)):
        raise ValueError(
            "Source CP offsets must own every modality row ordinal exactly once."
        )
    placeholder_counts = (global_input_ids == token_id).sum(dim=1).to(device="cpu")
    if source_attention_mask is None:
        source_attention_mask = (
            torch.arange(source_length).unsqueeze(0)
            < placeholder_counts.unsqueeze(1)
        )
    source_attention_mask = source_attention_mask.detach().to(
        device="cpu", dtype=torch.bool
    )
    if source_attention_mask.shape != (global_input_ids.shape[0], source_length):
        raise ValueError(
            "The modality attention mask must have shape (batch, projected_sequence)."
        )
    if not torch.equal(source_attention_mask.sum(dim=1), placeholder_counts):
        raise ValueError(
            "Each sample's valid projected modality rows must equal its placeholder count."
        )
    destination_flat = [
        position for rank_offsets in destination for position in rank_offsets
    ]
    if sorted(destination_flat) != list(range(seq_len)):
        raise ValueError(
            "Destination CP offsets must own every merged-text position exactly once."
        )

    destination_owner = {
        position: cp_rank
        for cp_rank, rank_offsets in enumerate(destination)
        for position in rank_offsets
    }
    rank_to_index = seam_group.rank_to_index

    source_cp = (
        seam_group.producer_ranks.index(global_rank)
        if global_rank in seam_group.producer_ranks
        else None
    )
    destination_cp = (
        seam_group.consumer_ranks.index(global_rank)
        if global_rank in seam_group.consumer_ranks
        else None
    )

    local_source_keys = (
        _placeholder_keys_by_source_ordinal(
            global_input_ids,
            token_id,
            source_attention_mask,
            source[source_cp],
        )
        if source_cp is not None
        else []
    )
    send_by_peer: list[list[int]] = [[] for _ in seam_group.ranks]
    for row, key in enumerate(local_source_keys):
        peer = seam_group.consumer_ranks[destination_owner[key[1]]]
        send_by_peer[rank_to_index[peer]].append(row)
    send_indices = tuple(row for rows in send_by_peer for row in rows)
    send_counts = tuple(len(rows) for rows in send_by_peer)
    local_source_width = len(source[source_cp]) if source_cp is not None else 0
    source_select_indices = tuple(
        batch_index * local_source_width + local_index
        for batch_index in range(global_input_ids.shape[0])
        for local_index, ordinal in enumerate(source[source_cp] if source_cp is not None else [])
        if bool(source_attention_mask[batch_index, ordinal])
    )

    desired_keys = (
        _placeholder_keys(global_input_ids, token_id, destination[destination_cp])
        if destination_cp is not None
        else []
    )
    incoming_by_peer: list[list[tuple[int, int]]] = [[] for _ in seam_group.ranks]
    if destination_cp is not None:
        # Reconstruct each source's exact packing order, then retain only rows
        # addressed to this destination. This keeps restoration correct even if
        # a certified splitter stores non-monotonic offsets.
        for source_cp_rank, source_rank in enumerate(seam_group.producer_ranks):
            source_keys = _placeholder_keys_by_source_ordinal(
                global_input_ids,
                token_id,
                source_attention_mask,
                source[source_cp_rank],
            )
            incoming_by_peer[rank_to_index[source_rank]].extend(
                key for key in source_keys
                if destination_owner[key[1]] == destination_cp
            )
    recv_counts = tuple(len(keys) for keys in incoming_by_peer)
    arrival_keys = [key for keys in incoming_by_peer for key in keys]
    arrival_index = {key: index for index, key in enumerate(arrival_keys)}
    restore_indices = tuple(arrival_index[key] for key in desired_keys)

    return RoutePlan(
        source_select_indices=source_select_indices,
        send_indices=send_indices,
        restore_indices=restore_indices,
        send_counts=send_counts,
        recv_counts=recv_counts,
        source_rows=len(local_source_keys),
        source_storage_rows=global_input_ids.shape[0] * local_source_width,
        destination_rows=len(desired_keys),
    )


def _pack(features: torch.Tensor, plan: RoutePlan) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1])
    if flat.shape[0] == plan.source_storage_rows:
        source_index = torch.tensor(
            plan.source_select_indices, dtype=torch.long, device=flat.device
        )
        flat = flat.index_select(0, source_index)
    elif flat.shape[0] != plan.source_rows:
        raise ValueError(
            "The modality projector must produce exactly one feature row per "
            f"locally owned placeholder; got {flat.shape[0]} rows for "
            f"{plan.source_rows} placeholders. Projectors with a different output "
            "length require an explicit placeholder span map."
        )
    index = torch.tensor(plan.send_indices, dtype=torch.long, device=flat.device)
    return flat.index_select(0, index)


def _restore(received: torch.Tensor, plan: RoutePlan) -> torch.Tensor:
    index = torch.tensor(plan.restore_indices, dtype=torch.long, device=received.device)
    return received.index_select(0, index)


def route_autograd(
    features: torch.Tensor,
    plan: RoutePlan,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Route local rows differentiably through one cross-mesh seam group."""
    packed = _pack(features, plan)
    received = _VariableSplitAllToAll.apply(
        packed, plan.recv_counts, plan.send_counts, group
    )
    return _restore(received, plan)


def exchange_forward(
    features: torch.Tensor,
    plan: RoutePlan,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Non-autograd forward used at an explicit pipeline graph break."""
    packed = _pack(features, plan)
    received = features.new_empty((sum(plan.recv_counts), features.shape[-1]))
    dist.all_to_all_single(
        received,
        packed.contiguous(),
        output_split_sizes=list(plan.recv_counts),
        input_split_sizes=list(plan.send_counts),
        group=group,
    )
    return _restore(received, plan)


def exchange_backward(
    destination_grad: torch.Tensor,
    state: GroupExchangeState,
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Transpose a pipeline seam exchange without a full gradient collective."""
    plan = state.plan
    if destination_grad.reshape(-1, hidden_size).shape[0] != plan.destination_rows:
        raise ValueError(
            "Destination feature gradient row count changed across the pipeline seam."
        )
    grad = destination_grad.reshape(-1, hidden_size).to(device=device, dtype=dtype)
    arrival_grad = grad.new_empty((sum(plan.recv_counts), hidden_size))
    if plan.restore_indices:
        restore = torch.tensor(plan.restore_indices, dtype=torch.long, device=device)
        arrival_grad.index_copy_(0, restore, grad)
    packed_grad = grad.new_empty((sum(plan.send_counts), hidden_size))
    dist.all_to_all_single(
        packed_grad,
        arrival_grad.contiguous(),
        output_split_sizes=list(plan.send_counts),
        input_split_sizes=list(plan.recv_counts),
        group=state.seam_group.process_group,
    )
    source_grad = grad.new_zeros((plan.source_rows, hidden_size))
    if plan.send_indices:
        send_index = torch.tensor(plan.send_indices, dtype=torch.long, device=device)
        source_grad.index_add_(0, send_index, packed_grad)
    return source_grad


class CrossMeshRouter:
    """Groups and rank-local routing for one encoder-to-language-model seam."""

    def __init__(self, groups: Sequence[CrossMeshGroup]) -> None:
        rank = dist.get_rank()
        self._groups = [group for group in groups if rank in group.ranks]
        self._rank = rank

    @property
    def groups(self) -> tuple[CrossMeshGroup, ...]:
        return tuple(self._groups)

    def forward(
        self,
        features: torch.Tensor | None,
        *,
        global_input_ids: torch.Tensor,
        token_id: int,
        source_attention_mask: torch.Tensor | None,
        source_offsets: Sequence[torch.Tensor],
        destination_offsets: Sequence[torch.Tensor],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, SeamExchangeState]:
        received: torch.Tensor | None = None
        states: list[GroupExchangeState] = []
        source_shape = features.shape if features is not None else None
        source_groups = 0

        for seam_group in self._groups:
            plan = build_route_plan(
                global_input_ids=global_input_ids,
                token_id=token_id,
                source_attention_mask=source_attention_mask,
                source_offsets=source_offsets,
                destination_offsets=destination_offsets,
                seam_group=seam_group,
                global_rank=self._rank,
            )
            is_source = self._rank in seam_group.producer_ranks
            is_destination = self._rank in seam_group.consumer_ranks
            if is_source:
                source_groups += 1
                if features is None:
                    raise ValueError("A producer seam rank did not supply modality features.")
                local = features.to(device=device, dtype=dtype)
            else:
                local = torch.empty((0, hidden_size), device=device, dtype=dtype)
            routed = exchange_forward(local, plan, seam_group.process_group)
            if is_destination:
                if received is not None:
                    raise RuntimeError("A destination rank belongs to multiple seam lanes.")
                received = routed
            states.append(GroupExchangeState(seam_group, plan))

        return received, SeamExchangeState(states, source_shape, source_groups)

    def backward(
        self,
        destination_grad: torch.Tensor | None,
        state: SeamExchangeState,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        source_grad: torch.Tensor | None = None
        for group_state in state.groups:
            group = group_state.seam_group
            if self._rank in group.consumer_ranks:
                if destination_grad is None:
                    raise ValueError("A consumer seam rank did not supply a feature gradient.")
                local_grad = destination_grad
            else:
                local_grad = torch.empty((0, hidden_size), device=device, dtype=dtype)
            returned = exchange_backward(
                local_grad,
                group_state,
                hidden_size=hidden_size,
                dtype=dtype,
                device=device,
            )
            if self._rank in group.producer_ranks:
                source_grad = returned if source_grad is None else source_grad + returned

        if source_grad is not None and state.source_fanout > 1:
            source_grad.div_(state.source_fanout)
        if source_grad is not None and state.source_shape is not None:
            plan = state.groups[0].plan
            storage_rows = state.source_shape.numel() // hidden_size
            if storage_rows == plan.source_storage_rows:
                storage_grad = source_grad.new_zeros((storage_rows, hidden_size))
                if plan.source_select_indices:
                    source_index = torch.tensor(
                        plan.source_select_indices,
                        dtype=torch.long,
                        device=device,
                    )
                    storage_grad.index_copy_(0, source_index, source_grad)
                source_grad = storage_grad
            elif storage_rows != plan.source_rows:
                raise ValueError(
                    "The producer feature shape changed across the pipeline seam."
                )
            source_grad = source_grad.reshape(state.source_shape)
        return source_grad


def build_cross_mesh_groups(
    *,
    producer_layout: Any,
    consumer_layout: Any,
) -> list[CrossMeshGroup]:
    """Collectively construct deterministic DP-local seam groups on all ranks.

    The projected hidden dimension is replicated today. Equal TP/EP degrees pair
    like lanes; a producer degree of one may fan out to replicated consumer
    lanes. A producer-sharded layout cannot be collapsed implicitly and is
    rejected rather than mixing incompatible hidden-dimension shards.
    """
    if producer_layout.dp_size != consumer_layout.dp_size:
        raise ValueError("Encoder and language-model seam layouts must have equal DP size.")
    for axis in ("tp_size", "ep_size"):
        producer_degree = getattr(producer_layout, axis)
        consumer_degree = getattr(consumer_layout, axis)
        if producer_degree not in (1, consumer_degree):
            label = axis.removesuffix("_size").upper()
            raise ValueError(
                f"Incompatible {label} layout at modality seam: producer degree "
                f"{producer_degree}, consumer degree {consumer_degree}. The "
                "projected hidden dimension must be replicated on the producer "
                "or paired across equal lanes."
            )

    groups: list[CrossMeshGroup] = []
    for dp_rank in range(producer_layout.dp_size):
        for consumer_tp in range(consumer_layout.tp_size):
            producer_tp = 0 if producer_layout.tp_size == 1 else consumer_tp
            for consumer_ep in range(consumer_layout.ep_size):
                producer_ep = 0 if producer_layout.ep_size == 1 else consumer_ep
                producer_ranks = tuple(
                    producer_layout.rank_at(
                        dp_rank,
                        producer_layout.last_stage,
                        cp_rank,
                        producer_tp,
                        producer_ep,
                    )
                    for cp_rank in range(producer_layout.cp_size)
                )
                consumer_ranks = tuple(
                    consumer_layout.rank_at(
                        dp_rank, 0, cp_rank, consumer_tp, consumer_ep
                    )
                    for cp_rank in range(consumer_layout.cp_size)
                )
                ranks = tuple(dict.fromkeys((*producer_ranks, *consumer_ranks)))
                process_group = dist.new_group(ranks=list(ranks))
                groups.append(
                    CrossMeshGroup(
                        ranks=ranks,
                        producer_ranks=producer_ranks,
                        consumer_ranks=consumer_ranks,
                        process_group=process_group,
                        dp_rank=dp_rank,
                        producer_tp_rank=producer_tp,
                        producer_ep_rank=producer_ep,
                        consumer_tp_rank=consumer_tp,
                        consumer_ep_rank=consumer_ep,
                    )
                )
    return groups


def routing_offsets_from_batch(
    batch: Mapping[str, Any], module_id: int
) -> Sequence[torch.Tensor]:
    """Read module-specific offsets preserved by ``ParallelContext``."""
    metadata = batch.get(CP_ROUTING_OFFSETS_KEY)
    if not isinstance(metadata, Mapping) or module_id not in metadata:
        raise ValueError(
            "Cross-mesh CP routing metadata is missing. Use "
            "ParallelContext.prepare_dataloader() so both meshes' actual offsets "
            "are preserved per microbatch."
        )
    return metadata[module_id]
