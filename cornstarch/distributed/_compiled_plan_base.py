"""Process-group-independent partition compilation and explicit activation.

This module is additive to the normal homogeneous materialization path.  It is
used by elastic orchestrators that must decide rank-local model ownership before
WORLD exists and repeatedly retire/activate different pipeline partitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from cornstarch.distributed.context_parallel import apply_context_parallel
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.parallelization import (
    MeshLayout,
    ParallelContext as _ParallelContext,
    ParallelizationPlan as _ParallelizationPlan,
)
from cornstarch.distributed.pipeline_parallel.forward_spec_wrapper import (
    PipelineParallelForwardSpec,
)
from cornstarch.distributed.tensor_parallel import apply_tensor_parallel
from cornstarch.models.language_model import CornstarchLanguageModel
from cornstarch.models.model_base import CornstarchModelBase
from cornstarch.models.multimodal.modeling import CornstarchModalityEncoder


@dataclass(frozen=True, slots=True)
class PipelineStageSpec:
    pipeline_id: str
    stage_id: int
    layer_start: int
    layer_end: int
    ranks: tuple[int, ...]
    is_first: bool = False
    is_last: bool = False
    tied_parameter_owners: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("pipeline_id must not be empty")
        if self.stage_id < 0 or self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("stage identity and global layer range are invalid")
        if not self.ranks or len(self.ranks) != len(set(self.ranks)):
            raise ValueError("stage ranks must be non-empty and unique")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("stage ranks must be non-negative")


@dataclass(frozen=True, slots=True)
class LogicalStateManifestEntry:
    logical_key: str
    global_shape: tuple[int, ...]
    dtype: str
    state_kind: str
    local_shape: tuple[int, ...]
    placements: tuple[str, ...]
    global_layer_id: int | None
    shared_state_id: str | None
    tp_lane: int


@dataclass(frozen=True, slots=True)
class LogicalStateManifest:
    rank: int
    entries: tuple[LogicalStateManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class _ModuleCompilation:
    module: CornstarchModelBase | CornstarchModalityEncoder
    config: Any
    stage: PipelineStageSpec
    layers_name: str
    blueprint_layers: tuple[nn.Module, ...]
    original_forward_spec: Any


@dataclass(frozen=True, slots=True)
class CompiledParallelizationPlan:
    world_size: int
    rank: int
    modules: tuple[_ModuleCompilation, ...]
    local_state_manifest: LogicalStateManifest
    _owner: "ParallelizationPlan" = field(repr=False, compare=False)

    def activate(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
        *,
        mesh: Any = None,
    ) -> "ParallelContext":
        return self._owner._activate_compiled(self, device=device, dtype=dtype, mesh=mesh)


class ParallelContext:
    """Closable proxy over the existing Cornstarch runtime context."""

    def __init__(
        self,
        context: _ParallelContext,
        *,
        compiled: CompiledParallelizationPlan | None = None,
        manifest: LogicalStateManifest | None = None,
    ) -> None:
        self._context = context
        self._compiled = compiled
        self.local_state_manifest = manifest
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._context, name)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        if self._compiled is not None:
            self._compiled._owner._restore_blueprint(self._compiled)
        for name in (
            "_meshes",
            "_layouts",
            "_cp_gradient_synchronizers",
            "_cross_mesh_groups",
        ):
            value = getattr(self._context, name, None)
            clear = getattr(value, "clear", None)
            if callable(clear):
                clear()
        self._context._gradient_synchronizer = None
        self._context._dp_group = None
        self._closed = True


class ParallelizationPlan(_ParallelizationPlan):
    """Parallelization plan with additive compile/activate lifecycle hooks."""

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> ParallelContext:
        # Compatibility wrapper: the inherited path is intentionally untouched.
        return ParallelContext(super().materialize(device=device, dtype=dtype))

    def compile(
        self,
        *,
        world_size: int,
        rank: int,
        stage_overrides: Sequence[PipelineStageSpec] | None = None,
    ) -> CompiledParallelizationPlan:
        if not self._configs:
            raise ValueError("No modules registered with parallelize().")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("world_size/rank are invalid")
        if stage_overrides is None:
            stage_overrides = self._default_stage_specs(world_size, rank)
        if len(stage_overrides) != len(self._modules):
            raise ValueError("stage_overrides must contain one spec per registered module")

        compilations = []
        expected_entries: list[LogicalStateManifestEntry] = []
        for module, config, stage in zip(self._modules, self._configs, stage_overrides):
            target = module.encoder if isinstance(module, CornstarchModalityEncoder) else module
            _, layers_name, _ = target._section_names()
            layers = getattr(target, layers_name)
            if stage.layer_end > len(layers):
                raise ValueError(
                    f"stage range [{stage.layer_start}, {stage.layer_end}) exceeds "
                    f"the {len(layers)} global repeated layers"
                )
            if rank not in stage.ranks:
                raise ValueError(f"rank {rank} is not an owner in stage {stage.pipeline_id}:{stage.stage_id}")
            compilation = _ModuleCompilation(
                module,
                config,
                stage,
                layers_name,
                tuple(layers),
                target.forward_spec,
            )
            compilations.append(compilation)
            expected_entries.extend(_compile_manifest_entries(target, compilation, rank))
        return CompiledParallelizationPlan(
            world_size,
            rank,
            tuple(compilations),
            LogicalStateManifest(rank, tuple(expected_entries)),
            self,
        )

    def _default_stage_specs(
        self, world_size: int, rank: int
    ) -> tuple[PipelineStageSpec, ...]:
        if any(config.uses_pipeline_parallel for config in self._configs):
            raise ValueError(
                "compile() requires explicit stage_overrides for pipeline-parallel modules"
            )
        specs = []
        for index, (module, config) in enumerate(zip(self._modules, self._configs)):
            target = module.encoder if isinstance(module, CornstarchModalityEncoder) else module
            layer_count = len(target._repeated_layers())
            tp = config.tensor_parallel_size
            start = (rank // tp) * tp
            specs.append(
                PipelineStageSpec(
                    f"module-{index}",
                    0,
                    0,
                    layer_count,
                    tuple(range(start, start + tp)),
                    True,
                    True,
                )
            )
        return tuple(specs)

    def _activate_compiled(
        self,
        compiled: CompiledParallelizationPlan,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None,
        mesh: Any,
    ) -> ParallelContext:
        device = torch.device(device)
        supplied = mesh if isinstance(mesh, Mapping) else None
        meshes: dict[int, Any] = {}
        layouts: dict[int, MeshLayout] = {}
        uses_pipeline = False

        for item in compiled.modules:
            module, config, stage = item.module, item.config, item.stage
            target = module.encoder if isinstance(module, CornstarchModalityEncoder) else module
            self._restore_module(item)
            local_mesh = supplied.get(id(module)) if supplied is not None else mesh
            needs_mesh = (
                config.tensor_parallel_size > 1
                or config.context_parallel_size > 1
                or config.expert_parallel_size > 1
                or not (stage.is_first and stage.is_last)
            )
            if needs_mesh and local_mesh is None:
                raise ValueError("activate() requires the Oobleck-created mesh for a distributed stage")
            if local_mesh is not None:
                meshes[id(module)] = local_mesh
                layouts[id(module)] = MeshLayout(
                    global_ranks=tuple(local_mesh._global_ranks),
                    dp_size=local_mesh.dp_size,
                    num_pp_stages=local_mesh.num_stages,
                    cp_size=local_mesh.cp_size,
                    tp_size=local_mesh.tp_size,
                    ep_size=local_mesh.ep_size,
                )
            if config.tensor_parallel_size > 1:
                apply_tensor_parallel(target, local_mesh.tp_mesh)
            if config.context_parallel_size > 1:
                apply_context_parallel(
                    target,
                    local_mesh.cp_group,
                    causal=isinstance(target, CornstarchLanguageModel),
                    splitter=config.context_parallel_splitter,
                )
            if not (stage.is_first and stage.is_last):
                _apply_explicit_pipeline_stage(target, item, local_mesh)
                uses_pipeline = True
            module.materialize(device, dtype=dtype)
            if config.expert_parallel_size > 1:
                apply_expert_parallel(target, local_mesh.ep_group)

        dp_size, dp_rank, dp_group, gradient_sync = self._build_dp_handles(meshes)
        cp_syncs = self._build_cp_gradient_synchronizers(meshes)
        cross_mesh = self._build_cross_mesh_groups(layouts) if layouts else {}
        context = _ParallelContext(
            modules=list(self._modules),
            configs=list(self._configs),
            meshes=meshes,
            layouts=layouts,
            dp_size=dp_size,
            dp_rank=dp_rank,
            dp_group=dp_group,
            gradient_synchronizer=gradient_sync,
            cp_gradient_synchronizers=cp_syncs,
            cross_mesh_groups=cross_mesh,
            uses_pipeline_parallel=uses_pipeline,
        )
        manifest = _activated_manifest(compiled)
        return ParallelContext(context, compiled=compiled, manifest=manifest)

    @staticmethod
    def _restore_module(item: _ModuleCompilation) -> None:
        target = item.module.encoder if isinstance(item.module, CornstarchModalityEncoder) else item.module
        setattr(target, item.layers_name, nn.ModuleList(item.blueprint_layers))
        target.forward_spec = item.original_forward_spec
        target._pipeline_layer_offset = 0
        target._local_tp_shard_specs = {}

    def _restore_blueprint(self, compiled: CompiledParallelizationPlan) -> None:
        for item in compiled.modules:
            self._restore_module(item)


def _apply_explicit_pipeline_stage(
    module: CornstarchModelBase,
    item: _ModuleCompilation,
    mesh: Any,
) -> None:
    start, end = item.stage.layer_start, item.stage.layer_end
    layers = getattr(module, item.layers_name)
    setattr(module, item.layers_name, nn.ModuleList(list(layers)[start:end]))
    module._pipeline_layer_offset = start
    local_specs = getattr(module, "_local_tp_shard_specs", {})
    prefix = f"{item.layers_name}."
    reindexed = {}
    for name, spec in local_specs.items():
        if not name.startswith(prefix):
            reindexed[name] = spec
            continue
        suffix = name[len(prefix):]
        index_text, separator, rest = suffix.partition(".")
        index = int(index_text)
        if start <= index < end:
            local = index - start
            reindexed[f"{prefix}{local}.{rest}" if separator else f"{prefix}{local}"] = spec
    module._local_tp_shard_specs = reindexed
    module.forward_spec = PipelineParallelForwardSpec(
        item.original_forward_spec, mesh, layer_offset=start
    )


def _compile_manifest_entries(
    module: CornstarchModelBase,
    item: _ModuleCompilation,
    rank: int,
) -> list[LogicalStateManifestEntry]:
    aliases: dict[int, str] = {}
    result = []
    for kind, values in (
        ("parameter", module.named_parameters(recurse=True, remove_duplicate=False)),
        ("buffer", module.named_buffers(recurse=True, remove_duplicate=False)),
    ):
        for name, tensor in values:
            prefix = f"{item.layers_name}."
            layer_id = None
            if name.startswith(prefix):
                index_text = name[len(prefix):].partition(".")[0]
                if index_text.isdigit():
                    layer_id = int(index_text)
                    if not item.stage.layer_start <= layer_id < item.stage.layer_end:
                        continue
            shared = aliases.setdefault(id(tensor), name)
            result.append(
                LogicalStateManifestEntry(
                    logical_key=name,
                    global_shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    state_kind=kind,
                    local_shape=tuple(tensor.shape),
                    placements=("pending-tp" if item.config.tensor_parallel_size > 1 else "replicate",),
                    global_layer_id=layer_id,
                    shared_state_id=shared if shared != name else None,
                    tp_lane=item.stage.ranks.index(rank),
                )
            )
    return result


def _activated_manifest(compiled: CompiledParallelizationPlan) -> LogicalStateManifest:
    expected = {entry.logical_key: entry for entry in compiled.local_state_manifest.entries}
    entries = []
    for item in compiled.modules:
        target = item.module.encoder if isinstance(item.module, CornstarchModalityEncoder) else item.module
        for kind, values in (
            ("parameter", target.named_parameters(recurse=True, remove_duplicate=False)),
            ("buffer", target.named_buffers(recurse=True, remove_duplicate=False)),
        ):
            for local_name, tensor in values:
                global_name = target._global_cornstarch_key(local_name)
                source = expected.get(global_name)
                if source is None:
                    continue
                local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                placements = tuple(str(value) for value in getattr(tensor, "placements", ())) or ("replicate",)
                entries.append(
                    LogicalStateManifestEntry(
                        global_name,
                        source.global_shape,
                        str(tensor.dtype),
                        kind,
                        tuple(local.shape),
                        placements,
                        source.global_layer_id,
                        source.shared_state_id,
                        source.tp_lane,
                    )
                )
    return LogicalStateManifest(compiled.rank, tuple(entries))
