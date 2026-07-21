"""Declarative per-module ``ParallelizationPlan``.

This is the ergonomic layer that most users touch.  They describe each
modality's parallel degrees with a :class:`ParallelConfig`, register the
modules with :meth:`ParallelizationPlan.parallelize`, then call
:meth:`ParallelizationPlan.materialize` once.  ``materialize`` returns a
:class:`ParallelContext` that folds the data-side parallelisms (DP sampler +
CP sequence split) into ``prepare_dataloader``, builds the training schedule,
and exposes gradient synchronization — so the training loop stays plain
PyTorch.

The plan is a thin orchestrator over the lower-level distributed primitives
(``ModalProcessGroupMesh`` + the four ``apply_*`` + ``GradientSynchronizer`` +
the schedule).  It hides the ordering rule (TP→CP→PP on the meta module, then
materialize, then EP on real tensors) and the per-modality + DP-offset + EP
rank math.  DP and CP stay data-side; TP/PP/EP stay model-side — the five
parallelisms remain independently togglable.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cornstarch.distributed.context_parallel import apply_context_parallel
from cornstarch.distributed.context_parallel.gated_delta import (
    build_gated_delta_metadata,
)
from cornstarch.distributed.context_parallel.splitters import ContextParallelSplitter
from cornstarch.distributed.cross_mesh_routing import (
    CP_MODALITY_MASKS_KEY,
    CP_ROUTING_OFFSETS_KEY,
    CrossMeshGroup,
    build_cross_mesh_groups,
)
from cornstarch.distributed.data_parallel import GradientSynchronizer
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.parallel_config import ParallelConfig
from cornstarch.distributed.pipeline_parallel import (
    PipelinePartitionSpec,
    apply_pipeline_parallel,
)
from cornstarch.distributed.pipeline_parallel.schedule import (
    MeshLayout,
    OneForwardOneBackwardSchedule,
    TrainingSchedule,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.tensor_parallel import apply_tensor_parallel
from cornstarch.models.model_base import CornstarchModelBase
from cornstarch.models.language_model import CornstarchLanguageModel
from cornstarch.models.multimodal.modeling import (
    CornstarchFusedModalityEncoder,
    CornstarchModalityEncoder,
)

if TYPE_CHECKING:
    from cornstarch.models.multimodal.execution import (
        CornstarchExecutionPlan,
        ExecutionFuture,
    )


_DEFAULT_CP_SPLIT_KEYS = (
    "input_ids",
    "labels",
    "shift_labels",
    "attention_mask",
    "position_ids",
    "document_ids",
)

ParallelModule = (
    CornstarchModelBase
    | CornstarchModalityEncoder
    | CornstarchFusedModalityEncoder
)
PipelinePartitions = (
    PipelinePartitionSpec
    | Mapping[str, PipelinePartitionSpec]
    | None
)


@dataclass(frozen=True)
class _ContextTarget:
    """One local module whose input ownership is context parallel."""

    module: ParallelModule
    config: ParallelConfig
    group: dist.ProcessGroup


class _ContextBatchTransform:
    """Apply every CP data transformation before model execution.

    Context parallelism owns tokens, not parameters. Consequently global
    positions, shifted labels, cross-mesh ownership, and rank-local sequence
    slices are derived here in the collate path. The model only receives the
    resulting local batch plus opaque token-mixer metadata injected by the CP
    backend during materialization.
    """

    def __init__(
        self,
        modules: Sequence[ParallelModule],
        configs: Sequence[ParallelConfig],
        targets: Sequence[_ContextTarget],
        cp_split_keys: Sequence[str],
        has_cross_mesh_seams: bool,
    ) -> None:
        self._modules = modules
        self._configs = configs
        self._targets = targets
        self._cp_split_keys = cp_split_keys
        self._has_cross_mesh_seams = has_cross_mesh_seams

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._add_global_sequence_metadata(batch)
        self._add_cross_mesh_offsets(batch)
        self._add_loss_metadata(batch)
        self._split_local_sequences(batch)
        return batch

    @staticmethod
    def _add_global_sequence_metadata(batch: dict[str, Any]) -> None:
        """Preserve facts that cannot be reconstructed after CP slicing."""
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
            return
        batch.setdefault("cp_global_input_ids", input_ids)
        batch.setdefault(
            "position_ids",
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1),
        )
        batch.setdefault("attention_mask", torch.ones_like(input_ids, dtype=torch.bool))

    def _add_cross_mesh_offsets(self, batch: dict[str, Any]) -> None:
        """Record both sides' ownership without moving activation rows."""
        global_ids = batch.get("cp_global_input_ids")
        text_mask = batch.get("attention_mask")
        if (
            not self._has_cross_mesh_seams
            or not isinstance(global_ids, torch.Tensor)
            or text_mask is None
        ):
            return

        modality_masks = batch.get(CP_MODALITY_MASKS_KEY, {})
        routing_offsets: dict[int, tuple[torch.Tensor, ...]] = {}
        for module, config in zip(self._modules, self._configs):
            routing_mask = text_mask
            if isinstance(module, CornstarchModalityEncoder):
                routing_mask = modality_masks.get(module.modality)
                if routing_mask is None:
                    # The schedule can derive a one-row-per-placeholder mask
                    # once the DAG supplies this modality's token id.
                    continue
            elif isinstance(module, CornstarchFusedModalityEncoder):
                splitter = config.context_parallel_splitter
                modality_offsets = {}
                for modality in module.modalities:
                    modality_mask = modality_masks.get(modality)
                    if modality_mask is None:
                        continue
                    offsets = (
                        [torch.arange(modality_mask.shape[1], dtype=torch.long)]
                        if splitter is None
                        else splitter.offsets_for_size(
                            modality_mask, config.context_parallel_size
                        )
                    )
                    modality_offsets[modality] = tuple(offsets)
                routing_offsets[id(module)] = modality_offsets
                continue
            splitter = config.context_parallel_splitter
            offsets = (
                [torch.arange(routing_mask.shape[1], dtype=torch.long)]
                if splitter is None
                else splitter.offsets_for_size(
                    routing_mask, config.context_parallel_size
                )
            )
            routing_offsets[id(module)] = tuple(offsets)
        batch[CP_ROUTING_OFFSETS_KEY] = routing_offsets

    @staticmethod
    def _add_loss_metadata(batch: dict[str, Any]) -> None:
        """Shift labels globally so CP run boundaries retain their next token."""
        labels = batch.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.ndim < 2:
            return
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = -100
        batch.setdefault("shift_labels", shift_labels)
        batch.setdefault("num_items_in_batch", (batch["shift_labels"] != -100).sum())

    def _split_local_sequences(self, batch: dict[str, Any]) -> None:
        """Compute each ownership layout from the full batch, then slice once."""
        source_batch = dict(batch)
        split_keys_seen: set[str] = set()
        for target in self._targets:
            module, config, cp_group = target.module, target.config, target.group
            splitter = config.context_parallel_splitter
            assert splitter is not None
            is_modality = isinstance(module, CornstarchModalityEncoder)
            mask = (
                source_batch.get(CP_MODALITY_MASKS_KEY, {}).get(module.modality)
                if is_modality
                else source_batch.get("attention_mask")
            )
            if mask is None:
                if is_modality:
                    # Modality processors may already own their sharding; the
                    # seam derives missing offsets later from placeholders.
                    continue
                input_ids = source_batch.get("input_ids")
                if input_ids is None:
                    continue
                mask = torch.ones_like(input_ids, dtype=torch.float32)

            offsets_per_rank = splitter.compute_offsets(mask, cp_group)
            if is_modality:
                continue
            self._add_token_mixer_metadata(
                batch, source_batch, module, offsets_per_rank, mask
            )
            for key in self._cp_split_keys:
                value = source_batch.get(key)
                if (
                    key not in split_keys_seen
                    and isinstance(value, torch.Tensor)
                    and value.ndim >= 2
                ):
                    batch[key] = splitter.split(value, cp_group)
                    split_keys_seen.add(key)

    @staticmethod
    def _add_token_mixer_metadata(
        batch: dict[str, Any],
        source_batch: dict[str, Any],
        module: ParallelModule,
        offsets_per_rank: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> None:
        """Describe recurrent runs for token mixers that cannot infer CP state."""
        layer_types = getattr(getattr(module, "hf_config", None), "layer_types", ())
        if "linear_attention" not in layer_types:
            return
        document_ids = source_batch.get("document_ids")
        if document_ids is None:
            document_ids = source_batch.get("cp_document_ids")
        batch["cp_sequence_metadata"] = build_gated_delta_metadata(
            offsets_per_rank,
            mask,
            document_ids=document_ids,
        )


@dataclass(frozen=True)
class ScheduleContext:
    """Public immutable inputs for a caller-supplied pipeline schedule."""

    plan: "CornstarchExecutionPlan"
    output_future: "ExecutionFuture"
    layouts: Mapping[int, MeshLayout]
    meshes: Mapping[int, ModalProcessGroupMesh]
    data_parallel_size: int
    cross_mesh_groups: Mapping[tuple[int, int], tuple[CrossMeshGroup, ...]]
    routing_splitters: Mapping[int, tuple[Any, int]]

    def create_default_1f1b(self) -> OneForwardOneBackwardSchedule:
        return OneForwardOneBackwardSchedule(
            self.plan,
            self.output_future,
            dict(self.layouts),
            dict(self.meshes),
            self.data_parallel_size,
            {
                key: list(groups)
                for key, groups in self.cross_mesh_groups.items()
            },
            dict(self.routing_splitters),
        )


class ParallelContext:
    """Runtime handle returned by :meth:`ParallelizationPlan.materialize`.

    Holds the per-modality meshes, the data-parallel sampler coordinates, the
    configured CP splitters, and the cross-modality gradient synchronizer.  Use
    it to wrap the dataloader (DP sampler + CP split), build a schedule, and
    sync gradients after ``backward()``.
    """

    def __init__(
        self,
        *,
        modules: list[ParallelModule],
        configs: list[ParallelConfig],
        meshes: dict[int, ModalProcessGroupMesh],
        layouts: dict[int, MeshLayout],
        dp_size: int,
        dp_rank: int,
        dp_group: Optional[dist.ProcessGroup],
        gradient_synchronizer: Optional[GradientSynchronizer],
        cp_gradient_synchronizers: Sequence[GradientSynchronizer] = (),
        cross_mesh_groups: dict[tuple[int, int], list[CrossMeshGroup]] | None = None,
        uses_pipeline_parallel: bool = False,
    ) -> None:
        self._modules = modules
        self._configs = configs
        self._meshes = meshes
        self._layouts = layouts
        self._dp_size = dp_size
        self._dp_rank = dp_rank
        self._dp_group = dp_group
        self._gradient_synchronizer = gradient_synchronizer
        self._cp_gradient_synchronizers = list(cp_gradient_synchronizers)
        self._cross_mesh_groups = dict(cross_mesh_groups or {})
        self._uses_pipeline_parallel = uses_pipeline_parallel

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def uses_pipeline_parallel(self) -> bool:
        """Whether modules are disaggregated into pipeline stages (vs co-located).

        True iff every registered ``ParallelConfig.pipeline_parallel_size`` is a
        positive int. When False, all modules are co-located on shared ranks and
        the training loop runs the plan directly (no schedule).
        """
        return self._uses_pipeline_parallel

    @property
    def dp_size(self) -> int:
        return self._dp_size

    @property
    def dp_rank(self) -> int:
        return self._dp_rank

    @property
    def dp_group(self) -> Optional[dist.ProcessGroup]:
        return self._dp_group

    def get_mesh(self, module: ParallelModule) -> Optional[ModalProcessGroupMesh]:
        """Return the process-group mesh built for a parallelized module."""
        return self._meshes.get(id(module))

    def get_splitter(
        self, module: ParallelModule
    ) -> Optional[ContextParallelSplitter]:
        """Return the configured CP splitter for a module."""
        for m, cfg in zip(self._modules, self._configs):
            if m is module:
                return cfg.context_parallel_splitter
        return None

    # ------------------------------------------------------------------
    # Data side: DP sampler + CP sequence split
    # ------------------------------------------------------------------

    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int | None = None,
        collate_fn: Optional[Callable[[list], "dict | list[dict]"]] = None,
        *,
        shuffle: bool = False,
        sampler: Any | None = None,
        batch_sampler: Any | None = None,
        per_microbatch_transform: Callable[[Any], Any] | None = None,
        microbatch_views: Callable[[Any], Iterable[dict[str, Any]]] | None = None,
        cp_split_keys: Sequence[str] = _DEFAULT_CP_SPLIT_KEYS,
        **loader_kwargs: Any,
    ) -> DataLoader:
        """Return a ``DataLoader`` with the DP sampler and CP split folded in.

        Data parallelism becomes a ``DistributedSampler`` over the ``dp`` axis
        (every rank within a replica sees the same shard); context parallelism
        becomes a ``collate_fn`` transform that slices the sequence dimension of
        ``cp_split_keys`` for the current CP rank.  The model is never touched —
        both parallelisms live entirely in the data pipeline.

        **Microbatches.** ``collate_fn`` may return either a single batch ``dict``
        or a ``list[dict]`` — the list of microbatches for one optimizer step.
        Returning a list is how the user controls microbatching: Cornstarch never
        splits modality tensors itself (only the user knows, e.g., how images map
        to samples). The loader always yields a ``list[dict]`` (a bare ``dict`` is
        wrapped as a single-element list), and the DP/CP transforms are applied to
        each microbatch independently. Pipeline schedules consume this list as
        their microbatches; without pipeline parallelism the training loop
        iterates it with gradient accumulation.

        **Structured views.** Pass ``microbatch_views`` for a caller-owned
        microbatch object. It must yield each mutable batch dictionary that needs
        Cornstarch's CP metadata/slicing (for example separate encoder and LLM
        views); the surrounding object is preserved and returned unchanged.

        **Multimodal CP seam metadata.** A collator whose modality projector
        emits padded context-sharded rows should include
        ``cp_modality_attention_masks`` as a mapping from modality name to its
        boolean ``(batch, projected_sequence)`` mask. The encoder splitter's
        actual offsets are computed from that modality-specific mask, while the
        LLM splitter's offsets come from the full text mask. If the mapping is
        omitted, the schedule derives a left-packed mask from each sample's
        placeholder count; non-left-packed or independently padded projectors
        must provide the explicit mask.
        """
        if batch_sampler is not None:
            if sampler is not None or shuffle:
                raise ValueError(
                    "batch_sampler is mutually exclusive with sampler and shuffle."
                )
            if batch_size is not None:
                raise ValueError("Omit batch_size when providing batch_sampler.")
        elif batch_size is None:
            raise ValueError("batch_size is required when batch_sampler is absent.")

        if sampler is None and batch_sampler is None and self._dp_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._dp_size,
                rank=self._dp_rank,
                shuffle=shuffle,
            )

        cp_targets = [
            _ContextTarget(m, cfg, self._meshes[id(m)].cp_group)
            for m, cfg in zip(self._modules, self._configs)
            if cfg.context_parallel_size > 1
            and cfg.context_parallel_splitter is not None
            and id(m) in self._meshes
        ]

        apply_cp_split = _ContextBatchTransform(
            self._modules,
            self._configs,
            cp_targets,
            cp_split_keys,
            has_cross_mesh_seams=bool(self._cross_mesh_groups),
        )

        def wrapped_collate(samples: list) -> list[Any]:
            collated = (
                collate_fn(samples)
                if collate_fn is not None
                else _default_collate(samples)
            )
            # Normalize to one optimizer-step list. A caller-owned view selector
            # lets structured microbatches expose distinct encoder/LLM dictionaries
            # without Cornstarch knowing the structure's concrete type.
            microbatches = collated if isinstance(collated, list) else [collated]
            result: list[Any] = []
            for microbatch in microbatches:
                if per_microbatch_transform is not None:
                    microbatch = per_microbatch_transform(microbatch)
                views = (
                    tuple(microbatch_views(microbatch))
                    if microbatch_views is not None
                    else (microbatch,)
                )
                if not views:
                    raise ValueError("A planned microbatch must expose at least one view.")
                for view in views:
                    if not isinstance(view, dict):
                        raise TypeError(
                            "Every context-parallel microbatch view must be a dict."
                        )
                    apply_cp_split(view)
                result.append(microbatch)
            return result

        common_kwargs = dict(
            dataset=dataset,
            collate_fn=wrapped_collate,
            **loader_kwargs,
        )
        if batch_sampler is not None:
            return DataLoader(batch_sampler=batch_sampler, **common_kwargs)
        return DataLoader(
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            **common_kwargs,
        )

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def create_schedule(
        self,
        plan: "CornstarchExecutionPlan",
        output_future: "ExecutionFuture",
        *,
        schedule_factory: Callable[[ScheduleContext], TrainingSchedule] | None = None,
    ) -> TrainingSchedule:
        """Return the pipeline-parallel training schedule for the plan.

        Only call this when pipeline parallelism is used (``uses_pipeline_parallel``);
        without it there are no stages and the training loop runs the plan directly
        (``output_future.execute()`` + ``backward()`` per microbatch). The returned
        :class:`OneForwardOneBackwardSchedule` drives the global pipeline — the
        modality encoder(s) as leading stage(s) feeding the language-model stages,
        with the encoder→language-model boundary crossed by the cross-mesh seam.

        Construction is cheap (DAG/stage analysis + a rank-local program, no
        collectives), so the caller may rebuild it per step. The number of
        microbatches comes from the list passed to ``schedule.step``.
        """
        if not self._uses_pipeline_parallel:
            raise ValueError(
                "create_schedule() requires pipeline parallelism. With no pipeline "
                "parallelism the modules are co-located; run the plan directly "
                "(output_future.execute() + backward()) per microbatch instead."
            )
        context = ScheduleContext(
            plan=plan,
            output_future=output_future,
            layouts=MappingProxyType(
                {id(module): self._layouts[id(module)] for module in self._modules}
            ),
            meshes=MappingProxyType(dict(self._meshes)),
            data_parallel_size=self._dp_size,
            cross_mesh_groups=MappingProxyType(
                {
                    key: tuple(groups)
                    for key, groups in self._cross_mesh_groups.items()
                }
            ),
            routing_splitters=MappingProxyType(
                {
                    id(module): (
                        config.context_parallel_splitter,
                        config.context_parallel_size,
                    )
                    for module, config in zip(self._modules, self._configs)
                }
            ),
        )
        if schedule_factory is not None:
            return schedule_factory(context)
        return context.create_default_1f1b()

    # ------------------------------------------------------------------
    # Gradient sync
    # ------------------------------------------------------------------

    def sync_gradients(self) -> None:
        """Synchronize CP partial gradients, then average DP replicas.

        CP ranks own disjoint query/loss tokens, so their replicated parameter
        gradients are summed first. DP replicas are then averaged. Both groups
        fix the EP coordinate, so corresponding local expert shards are synced
        just like dense parameters.
        """
        for synchronizer in self._cp_gradient_synchronizers:
            synchronizer.sync()
        if self._gradient_synchronizer is not None:
            self._gradient_synchronizer.sync()


def _default_collate(samples: list) -> dict:
    """Stack a list of dict samples into a batch dict (tensors stacked dim 0)."""
    if not samples:
        return {}
    keys = samples[0].keys()
    batch: dict[str, Any] = {}
    for key in keys:
        values = [s[key] for s in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        else:
            batch[key] = torch.tensor(values)
    return batch


class ParallelizationPlan:
    """Declarative per-modality parallelism plan (Option C entry point).

    Usage::

        plan = ParallelizationPlan(global_ranks=list(range(world_size)))
        plan.parallelize(vision_encoder, ParallelConfig(tensor_parallel_size=2,
                                                        data_parallel_size=dp))
        plan.parallelize(language_model, ParallelConfig(
            tensor_parallel_size=2, pipeline_parallel_size=2,
            context_parallel_size=2, data_parallel_size=dp,
            context_parallel_splitter=UniformContextParallelSplitter(),
        ))

        ctx = plan.materialize(device, dtype=torch.bfloat16)
        loader = ctx.prepare_dataloader(dataset, batch_size=16, collate_fn=collate)
        schedule = ctx.create_schedule(exec_plan, output_future,
                                       num_microbatches=8, microbatch_size=2)
        for batch in loader:
            schedule.step(batch, criterion, optimizer, return_loss=True)
            ctx.sync_gradients(); optimizer.step(); optimizer.zero_grad()
    """

    def __init__(self, global_ranks: Optional[Iterable[int]] = None) -> None:
        self._modules: list[ParallelModule] = []
        self._configs: list[ParallelConfig] = []
        self._pipeline_partitions: list[PipelinePartitions] = []
        self._global_ranks: Optional[list[int]] = (
            list(global_ranks) if global_ranks is not None else None
        )

    def parallelize(
        self,
        module: ParallelModule,
        config: ParallelConfig,
        *,
        pipeline_partitions: PipelinePartitions = None,
    ) -> None:
        """Record a module-to-config binding for later distribution.

        Only Cornstarch models and ``CornstarchModalityEncoder`` units are
        accepted. A bare Hugging Face encoder (or any other module) is rejected:
        it has no projector lifecycle and no ``_section_names()`` for the
        ``apply_*`` helpers to walk. Wrap such an encoder with
        ``build_modality_encoder(encoder, language_model, modality=...)`` first.
        """
        if not isinstance(
            module,
            (
                CornstarchModelBase,
                CornstarchModalityEncoder,
                CornstarchFusedModalityEncoder,
            ),
        ):
            raise TypeError(
                f"parallelize() requires a CornstarchModelBase, "
                f"CornstarchModalityEncoder, or CornstarchFusedModalityEncoder; "
                f"got {type(module).__name__}. Wrap a "
                f"raw Hugging Face encoder with build_modality_encoder(encoder, "
                f"language_model, modality=...) before parallelizing it."
            )
        if pipeline_partitions is not None and not config.uses_pipeline_parallel:
            raise ValueError(
                "pipeline_partitions requires a positive pipeline_parallel_size."
            )
        if isinstance(module, CornstarchFusedModalityEncoder):
            if pipeline_partitions is not None:
                if not isinstance(pipeline_partitions, Mapping):
                    raise TypeError(
                        "A fused module requires a modality-to-partition mapping."
                    )
                expected = set(module.modalities)
                actual = set(pipeline_partitions)
                if actual != expected:
                    raise ValueError(
                        "Fused partition keys must exactly match registered "
                        f"modalities; expected {sorted(expected)}, got {sorted(actual)}."
                    )
                if any(
                    len(spec.boundaries) != config.num_pp_stages
                    for spec in pipeline_partitions.values()
                ):
                    raise ValueError(
                        "Every fused partition must contain one boundary per PP stage."
                    )
        elif isinstance(pipeline_partitions, Mapping):
            raise TypeError("A non-fused module accepts one PipelinePartitionSpec.")
        elif (
            pipeline_partitions is not None
            and len(pipeline_partitions.boundaries) != config.num_pp_stages
        ):
            raise ValueError("Partition must contain one boundary per PP stage.")

        self._modules.append(module)
        self._configs.append(config)
        self._pipeline_partitions.append(pipeline_partitions)

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> ParallelContext:
        """Apply all parallelism strategies and materialize parameters.

        For each registered module, in order: builds its ``ModalProcessGroupMesh``,
        applies TP (DTensor weight sharding), CP (attention injection), and PP
        (forward-spec wrapping + layer slicing) on the meta module, materializes
        it (optionally in ``dtype``), then applies EP (slicing the materialized
        expert tensors and injecting all-to-all dispatch over the EP mesh axis).
        Returns a :class:`ParallelContext` for the training loop.
        """
        if not self._configs:
            raise ValueError("No modules registered with parallelize().")

        device = torch.device(device)
        global_ranks = self._resolve_global_ranks()
        pipelined, dp_size, module_ranks = self._assign_ranks(
            global_ranks, len(global_ranks)
        )
        self._validate_dp_size(dp_size)

        meshes: dict[int, ModalProcessGroupMesh] = {}
        layouts: dict[int, MeshLayout] = {}
        for module, config, ranks, partitions in zip(
            self._modules,
            self._configs,
            module_ranks,
            self._pipeline_partitions,
        ):
            layout, mesh = self._build_module_mesh(
                config, ranks, dp_size, device.type
            )
            layouts[id(module)] = layout
            if not mesh.is_member:
                continue
            meshes[id(module)] = mesh
            self._materialize_local_module(
                module,
                config,
                mesh,
                device,
                dtype,
                pipeline_partitions=partitions,
            )

        dp_size_final, dp_rank, dp_group, grad_sync = self._build_dp_handles(
            meshes
        )
        cp_grad_syncs = self._build_cp_gradient_synchronizers(meshes)
        cross_mesh_groups = self._build_cross_mesh_groups(layouts)

        return ParallelContext(
            modules=self._modules,
            configs=self._configs,
            meshes=meshes,
            layouts=layouts,
            dp_size=dp_size_final,
            dp_rank=dp_rank,
            dp_group=dp_group,
            gradient_synchronizer=grad_sync,
            cp_gradient_synchronizers=cp_grad_syncs,
            cross_mesh_groups=cross_mesh_groups,
            uses_pipeline_parallel=pipelined,
        )

    def _resolve_global_ranks(self) -> list[int]:
        """Return a permutation of the whole world, rejecting partial plans.

        Per-module grids may be distinct or overlap when co-located, but one
        ``ParallelizationPlan`` is the authority for the current distributed
        job. Silently omitting a world rank would leave that process outside the
        deterministic process-group creation order and can deadlock later.
        """
        world_size = dist.get_world_size()
        ranks = (
            list(self._global_ranks)
            if self._global_ranks is not None
            else list(range(world_size))
        )
        if sorted(ranks) != list(range(world_size)):
            raise ValueError(
                "ParallelizationPlan.global_ranks must contain every world rank "
                f"exactly once; expected 0..{world_size - 1}, got {ranks}."
            )
        return ranks

    def _validate_dp_size(self, computed_dp_size: int) -> None:
        for config in self._configs:
            if config.data_parallel_size != computed_dp_size:
                raise ValueError(
                    f"ParallelConfig.data_parallel_size={config.data_parallel_size} "
                    f"does not match the computed dp_size={computed_dp_size}."
                )

    @staticmethod
    def _build_module_mesh(
        config: ParallelConfig,
        ranks: list[int],
        dp_size: int,
        device_type: str,
    ) -> tuple[MeshLayout, ModalProcessGroupMesh]:
        """Describe and collectively construct one module's five-axis grid.

        Every world rank calls this helper for every module in registration
        order. ``DeviceMesh`` creates process groups collectively, so skipping a
        non-local module here would make ranks disagree on collective ordering.
        The pure ``MeshLayout`` is retained everywhere for DAG seam routing;
        only member ranks keep the operational mesh.
        """
        layout = MeshLayout(
            global_ranks=tuple(ranks),
            dp_size=dp_size,
            num_pp_stages=config.num_pp_stages,
            cp_size=config.context_parallel_size,
            tp_size=config.tensor_parallel_size,
            ep_size=config.expert_parallel_size,
        )
        mesh = ModalProcessGroupMesh(
            device_type=device_type,
            global_ranks=ranks,
            dp_size=dp_size,
            cp_size=config.context_parallel_size,
            tp_size=config.tensor_parallel_size,
            num_pp_stages=config.num_pp_stages,
            ep_size=config.expert_parallel_size,
        )
        return layout, mesh

    @staticmethod
    def _materialize_local_module(
        module: ParallelModule,
        config: ParallelConfig,
        mesh: ModalProcessGroupMesh,
        device: torch.device,
        dtype: torch.dtype | None,
        *,
        pipeline_partitions: PipelinePartitions = None,
    ) -> None:
        """Apply model-side axes around rank-local lazy materialization.

        TP and PP change parameter ownership while tensors are still metadata;
        CP injects token-mixer functions but does not own parameters. Only then
        does ``materialize`` allocate/load this rank's tensors. EP slices real
        expert stacks afterward because its dispatcher operates on concrete
        batched weights. This ordering is the central lazy-initialization
        invariant and is intentionally expressed once.
        """
        if isinstance(module, CornstarchFusedModalityEncoder):
            partition_map = dict(pipeline_partitions or {})
            targets = [
                (name, child.encoder, partition_map.get(name))
                for name, child in module.encoders.items()
            ]
        else:
            target = (
                module.encoder
                if isinstance(module, CornstarchModalityEncoder)
                else module
            )
            partition = (
                pipeline_partitions
                if isinstance(pipeline_partitions, PipelinePartitionSpec)
                else None
            )
            targets = [(None, target, partition)]

        for _, target, partition in targets:
            if config.tensor_parallel_size > 1:
                apply_tensor_parallel(target, mesh.tp_mesh)
            if config.context_parallel_size > 1:
                apply_context_parallel(
                    target,
                    mesh.cp_group,
                    causal=isinstance(target, CornstarchLanguageModel),
                    splitter=config.context_parallel_splitter,
                )
            if config.num_pp_stages > 1:
                apply_pipeline_parallel(target, mesh, partition=partition)

        if isinstance(module, CornstarchFusedModalityEncoder):
            for child in module.encoders.values():
                child._pipeline_mesh = mesh
        elif isinstance(module, CornstarchModalityEncoder):
            module._pipeline_mesh = mesh

        module.materialize(device, dtype=dtype)
        if config.expert_parallel_size > 1:
            for _, target, _ in targets:
                apply_expert_parallel(target, mesh.ep_group)

    def _build_cross_mesh_groups(
        self, layouts: dict[int, MeshLayout]
    ) -> dict[tuple[int, int], list[CrossMeshGroup]]:
        """Build every encoder->LLM seam group in registration order.

        ``dist.new_group`` creation is a world-order-sensitive operation. Every
        rank executes these nested loops over the identical module list, even
        when it belongs to neither side of a particular seam.
        """
        encoders = [
            module
            for module in self._modules
            if isinstance(
                module,
                (CornstarchModalityEncoder, CornstarchFusedModalityEncoder),
            )
        ]
        language_models = [
            module for module in self._modules
            if isinstance(module, CornstarchLanguageModel)
        ]
        result: dict[tuple[int, int], list[CrossMeshGroup]] = {}
        for encoder in encoders:
            for language_model in language_models:
                result[(id(encoder), id(language_model))] = build_cross_mesh_groups(
                    producer_layout=layouts[id(encoder)],
                    consumer_layout=layouts[id(language_model)],
                )
        return result

    def _assign_ranks(
        self, global_ranks: list[int], world_size: int
    ) -> tuple[bool, int, list[list[int]]]:
        """Classify the PP intent and assign each module its global ranks.

        Returns ``(pipelined, dp_size, module_ranks)`` where ``module_ranks[i]``
        is the replica-major rank list for module ``i``.

        - **All** ``pipeline_parallel_size is None`` → not pipelined: every module
          is **co-located** on the *same* replica rank range (each rank runs every
          modality). Requires equal ``ranks_per_replica`` across modules.
        - **All** positive ints → pipelined: modules are **disaggregated** onto
          disjoint rank ranges (``sum(ranks_per_replica)`` per replica), as before.
        - A mix is rejected.
        """
        pp_flags = [cfg.uses_pipeline_parallel for cfg in self._configs]
        if all(pp_flags):
            pipelined = True
        elif not any(pp_flags):
            pipelined = False
        else:
            named = ", ".join(
                f"{type(m).__name__}(pipeline_parallel_size="
                f"{c.pipeline_parallel_size})"
                for m, c in zip(self._modules, self._configs)
            )
            raise ValueError(
                "All registered modules must agree on pipeline parallelism: "
                "either every ParallelConfig.pipeline_parallel_size is None "
                "(co-located, no pipeline parallelism) or every one is a positive "
                f"int (disaggregated, pipeline parallelism). Got a mix: {named}."
            )

        if pipelined:
            ranks_per_replica = sum(cfg.ranks_per_replica for cfg in self._configs)
            if world_size % ranks_per_replica != 0:
                raise ValueError(
                    f"world_size ({world_size}) is not divisible by the sum of "
                    f"per-modality ranks_per_replica ({ranks_per_replica})."
                )
            dp_size = world_size // ranks_per_replica
            sizes = [cfg.ranks_per_replica for cfg in self._configs]
            module_ranks: list[list[int]] = []
            for mod_idx in range(len(self._configs)):
                size = sizes[mod_idx]
                offset = sum(sizes[:mod_idx])
                ranks: list[int] = []
                for replica in range(dp_size):
                    base = replica * ranks_per_replica + offset
                    ranks.extend(global_ranks[base : base + size])
                module_ranks.append(ranks)
            return pipelined, dp_size, module_ranks

        # Co-located: all modules share one replica rank range.
        distinct = {cfg.ranks_per_replica for cfg in self._configs}
        if len(distinct) != 1:
            named = ", ".join(
                f"{type(m).__name__}={c.ranks_per_replica}"
                for m, c in zip(self._modules, self._configs)
            )
            raise ValueError(
                "Co-located modules (pipeline_parallel_size=None) must have equal "
                f"ranks_per_replica (tp*cp*ep); got {named}. Use equal tp/cp/ep, "
                "or set a positive pipeline_parallel_size to disaggregate them."
            )
        ranks_per_replica = distinct.pop()
        if world_size % ranks_per_replica != 0:
            raise ValueError(
                f"world_size ({world_size}) is not divisible by the shared "
                f"co-located ranks_per_replica ({ranks_per_replica})."
            )
        dp_size = world_size // ranks_per_replica
        # Every module spans the full replica layout (all ranks).
        module_ranks = [list(global_ranks) for _ in self._configs]
        return pipelined, dp_size, module_ranks

    def _build_dp_handles(
        self, meshes: dict[int, ModalProcessGroupMesh]
    ) -> tuple[int, int, Optional[dist.ProcessGroup], Optional[GradientSynchronizer]]:
        """Derive DP size/rank/group and build the gradient synchronizer."""
        first_mesh = next(
            (meshes[id(m)] for m in self._modules if id(m) in meshes), None
        )
        if first_mesh is None:
            return 1, 0, None, None

        dp_group = first_mesh.dp_group
        dp_size = first_mesh.dp_size
        try:
            dp_rank = first_mesh.device_mesh.get_local_rank("dp")
        except Exception:
            dp_rank = dist.get_rank(dp_group) if dp_size > 1 else 0

        grad_sync = GradientSynchronizer(
            dp_group,
            skip_expert_parallel=False,
        )
        for module in self._modules:
            if id(module) in meshes:
                module._dp_group = dp_group
                grad_sync.register(module)
        return dp_size, dp_rank, dp_group, grad_sync

    def _build_cp_gradient_synchronizers(
        self, meshes: dict[int, ModalProcessGroupMesh]
    ) -> list[GradientSynchronizer]:
        """Build one sum-reduction synchronizer per local CP module shard."""
        synchronizers: list[GradientSynchronizer] = []
        for module, config in zip(self._modules, self._configs):
            mesh = meshes.get(id(module))
            if mesh is None or config.context_parallel_size <= 1:
                continue
            synchronizer = GradientSynchronizer(
                mesh.cp_group,
                average=False,
                skip_expert_parallel=False,
            )
            synchronizer.register(module)
            synchronizers.append(synchronizer)
        return synchronizers
