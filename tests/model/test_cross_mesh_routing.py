from __future__ import annotations

import torch

from cornstarch.distributed.context_parallel.splitters import (
    ContextParallelSplitter,
    HeadTailContextParallelSplitter,
    UniformContextParallelSplitter,
)
from cornstarch.distributed.cross_mesh_routing import (
    CrossMeshGroup,
    build_cross_mesh_groups,
    build_route_plan,
)
from cornstarch.distributed.pipeline_parallel.schedule import MeshLayout


def _group(source: tuple[int, ...], destination: tuple[int, ...]) -> CrossMeshGroup:
    return CrossMeshGroup(
        ranks=tuple(dict.fromkeys((*source, *destination))),
        producer_ranks=source,
        consumer_ranks=destination,
        process_group=None,  # type: ignore[arg-type]
        dp_rank=0,
        producer_tp_rank=0,
        producer_ep_rank=0,
        consumer_tp_rank=0,
        consumer_ep_rank=0,
    )


def test_acceptance_route_vision_cp2_to_llm_cp4() -> None:
    token = 99
    ids = torch.arange(16).unsqueeze(0)
    ids[:, 4:10] = token
    modality_mask = torch.ones(1, 6, dtype=torch.bool)
    source = UniformContextParallelSplitter().offsets_for_size(modality_mask, 2)
    destination = UniformContextParallelSplitter().offsets_for_size(ids, 4)
    group = _group((0, 1), (2, 3, 4, 5))

    source0 = build_route_plan(
        global_input_ids=ids,
        token_id=token,
        source_attention_mask=modality_mask,
        source_offsets=source,
        destination_offsets=destination,
        seam_group=group,
        global_rank=0,
    )
    source1 = build_route_plan(
        global_input_ids=ids,
        token_id=token,
        source_attention_mask=modality_mask,
        source_offsets=source,
        destination_offsets=destination,
        seam_group=group,
        global_rank=1,
    )
    llm0 = build_route_plan(
        global_input_ids=ids,
        token_id=token,
        source_attention_mask=modality_mask,
        source_offsets=source,
        destination_offsets=destination,
        seam_group=group,
        global_rank=2,
    )

    assert source0.send_counts == (0, 0, 0, 3, 0, 0)
    assert source1.send_counts == (0, 0, 0, 1, 2, 0)
    assert llm0.destination_rows == 0
    assert sum(llm0.recv_counts) == 0


def test_variable_sample_counts_select_padded_source_rows() -> None:
    token = 9
    ids = torch.tensor(
        [
            [1, token, token, 2, 3, 4, token, 5],
            [token, 1, 2, 3, 4, token, 6, 7],
        ]
    )
    modality_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    source = UniformContextParallelSplitter().offsets_for_size(modality_mask, 2)
    destination = HeadTailContextParallelSplitter().offsets_for_size(ids, 2)
    group = _group((0, 1), (2, 3))

    source_tail = build_route_plan(
        global_input_ids=ids,
        token_id=token,
        source_attention_mask=modality_mask,
        source_offsets=source,
        destination_offsets=destination,
        seam_group=group,
        global_rank=1,
    )

    # CP1 owns modality ordinal 2. It is valid only in sample 0, while its
    # projector output still has one padded slot for sample 1.
    assert source_tail.source_storage_rows == 2
    assert source_tail.source_rows == 1
    assert source_tail.source_select_indices == (0,)
    assert sum(source_tail.send_counts) == 1


def test_route_rejects_projected_mask_placeholder_mismatch() -> None:
    token = 9
    ids = torch.tensor([[token, token, 1, 2]])
    bad_mask = torch.tensor([[1, 0]], dtype=torch.bool)
    offsets = UniformContextParallelSplitter().offsets_for_size(bad_mask, 1)
    group = _group((0,), (1,))

    try:
        build_route_plan(
            global_input_ids=ids,
            token_id=token,
            source_attention_mask=bad_mask,
            source_offsets=offsets,
            destination_offsets=(torch.arange(4),),
            seam_group=group,
            global_rank=0,
        )
    except ValueError as error:
        assert "placeholder count" in str(error)
    else:
        raise AssertionError("expected one-row-per-placeholder validation failure")


def test_legacy_splitter_subclass_retains_compute_offsets_contract() -> None:
    class LegacySplitter(ContextParallelSplitter):
        def compute_offsets(self, attention_mask, cp_group):
            self._offsets_per_rank = [torch.arange(attention_mask.shape[1])]
            return self._offsets_per_rank

    splitter = LegacySplitter()
    offsets = splitter.compute_offsets(torch.ones(1, 3), None)
    assert torch.equal(offsets[0], torch.arange(3))
    try:
        splitter.offsets_for_size(torch.ones(1, 3), 1)
    except NotImplementedError as error:
        assert "cross-mesh routing" in str(error)
    else:
        raise AssertionError("legacy splitter should require explicit routing support")


def test_offsets_for_size_rejects_nonpositive_cp_size() -> None:
    try:
        UniformContextParallelSplitter().offsets_for_size(torch.ones(1, 3), 0)
    except ValueError as error:
        assert "cp_size" in str(error)
    else:
        raise AssertionError("expected cp_size validation failure")


def test_incompatible_producer_hidden_shards_are_rejected() -> None:
    producer = MeshLayout((0, 1), 1, 1, 1, 2, 1)
    consumer = MeshLayout((2,), 1, 1, 1, 1, 1)
    try:
        build_cross_mesh_groups(
            producer_layout=producer, consumer_layout=consumer
        )
    except ValueError as error:
        assert "Incompatible TP layout" in str(error)
    else:
        raise AssertionError("expected incompatible TP seam validation failure")
