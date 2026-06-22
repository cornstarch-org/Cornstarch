"""Option C surface: declarative per-modality ``ParallelizationPlan``.

This is the ergonomic layer that most users touch.  They describe each
modality's parallel degrees with a :class:`ParallelConfig`, register the
modules with :meth:`ParallelizationPlan.parallelize`, then call
:meth:`ParallelizationPlan.materialize` once.  ``materialize`` returns a
:class:`ParallelContext` that folds the data-side parallelisms (DP sampler +
CP sequence split) into ``prepare_dataloader``, builds the training schedule,
and exposes gradient synchronization — so the training loop stays plain
PyTorch.

The plan is a thin orchestrator over the Option B primitives
(``ModalProcessGroupMesh`` + the four ``apply_*`` + ``GradientSynchronizer`` +
the schedule).  It hides the ordering rule (TP→CP→PP on the meta module, then
materialize, then EP on real tensors) and the per-modality + DP-offset + EP
rank math.  DP and CP stay data-side; TP/PP/EP stay model-side — the five
parallelisms remain independently togglable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from cornstarch.distributed.context_parallel import apply_context_parallel
from cornstarch.distributed.context_parallel.splitters import ContextParallelSplitter
from cornstarch.distributed.data_parallel import GradientSynchronizer
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.parallel_config import ParallelConfig
from cornstarch.distributed.pipeline_parallel import apply_pipeline_parallel
from cornstarch.distributed.pipeline_parallel.schedule import (
    MeshLayout,
    OneForwardOneBackwardSchedule,
    TrainingSchedule,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.tensor_parallel import apply_tensor_parallel
from cornstarch.models.model_base import CornstarchModelBase
from cornstarch.models.multimodal.modeling import CornstarchModalityEncoder

if TYPE_CHECKING:
    from cornstarch.models.multimodal.execution import (
        CornstarchExecutionPlan,
        ExecutionFuture,
    )


_DEFAULT_CP_SPLIT_KEYS = ("input_ids", "labels", "attention_mask", "position_ids")


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
        modules: list[CornstarchModelBase],
        configs: list[ParallelConfig],
        meshes: dict[int, ModalProcessGroupMesh],
        layouts: dict[int, MeshLayout],
        dp_size: int,
        dp_rank: int,
        dp_group: Optional[dist.ProcessGroup],
        gradient_synchronizer: Optional[GradientSynchronizer],
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

    def get_mesh(self, module: CornstarchModelBase) -> Optional[ModalProcessGroupMesh]:
        """Return the process-group mesh built for a parallelized module."""
        return self._meshes.get(id(module))

    def get_splitter(
        self, module: CornstarchModelBase
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
        batch_size: int,
        collate_fn: Optional[Callable[[list], "dict | list[dict]"]] = None,
        *,
        shuffle: bool = False,
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
        """
        sampler = None
        if self._dp_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self._dp_size,
                rank=self._dp_rank,
                shuffle=shuffle,
            )

        cp_targets = [
            (cfg.context_parallel_splitter, self._meshes[id(m)].cp_group)
            for m, cfg in zip(self._modules, self._configs)
            if cfg.context_parallel_size > 1
            and cfg.context_parallel_splitter is not None
            and id(m) in self._meshes
        ]

        def apply_cp_split(batch: dict) -> dict:
            for splitter, cp_group in cp_targets:
                mask = batch.get("attention_mask")
                if mask is None:
                    ref = batch.get("input_ids")
                    if ref is None:
                        continue
                    mask = torch.ones_like(ref, dtype=torch.float32)
                splitter.compute_offsets(mask, cp_group)
                for key in cp_split_keys:
                    value = batch.get(key)
                    if isinstance(value, torch.Tensor) and value.ndim >= 2:
                        batch[key] = splitter.split(value, cp_group)
            return batch

        def wrapped_collate(samples: list) -> list[dict]:
            collated = (
                collate_fn(samples)
                if collate_fn is not None
                else _default_collate(samples)
            )
            # Normalize to a microbatch list (a bare dict = a single microbatch),
            # then apply the DP/CP transforms per microbatch.
            microbatches = collated if isinstance(collated, list) else [collated]
            return [apply_cp_split(mb) for mb in microbatches]

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            collate_fn=wrapped_collate,
            **loader_kwargs,
        )

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def create_schedule(
        self,
        plan: "CornstarchExecutionPlan",
        output_future: "ExecutionFuture",
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
        return OneForwardOneBackwardSchedule(
            plan,
            output_future,
            {id(module): self._layouts[id(module)] for module in self._modules},
            self._meshes,
            self._dp_size,
        )

    # ------------------------------------------------------------------
    # Gradient sync
    # ------------------------------------------------------------------

    def sync_gradients(self) -> None:
        """All-reduce gradients across DP ranks for every registered module.

        Call after ``loss.backward()`` and before ``optimizer.step()``.
        Expert-parallel-sharded parameters are skipped automatically.
        """
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
        self._modules: list[CornstarchModelBase] = []
        self._configs: list[ParallelConfig] = []
        self._global_ranks: Optional[list[int]] = (
            list(global_ranks) if global_ranks is not None else None
        )

    def parallelize(
        self,
        module: CornstarchModelBase | CornstarchModalityEncoder,
        config: ParallelConfig,
    ) -> None:
        """Record a module-to-config binding for later distribution.

        Only Cornstarch models and ``CornstarchModalityEncoder`` units are
        accepted. A bare Hugging Face encoder (or any other module) is rejected:
        it has no projector lifecycle and no ``_section_names()`` for the
        ``apply_*`` helpers to walk. Wrap such an encoder with
        ``build_modality_encoder(encoder, language_model, modality=...)`` first.
        """
        if not isinstance(module, (CornstarchModelBase, CornstarchModalityEncoder)):
            raise TypeError(
                f"parallelize() requires a CornstarchModelBase or "
                f"CornstarchModalityEncoder, got {type(module).__name__}. Wrap a "
                f"raw Hugging Face encoder with build_modality_encoder(encoder, "
                f"language_model, modality=...) before parallelizing it."
            )
        self._modules.append(module)
        self._configs.append(config)

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
        device = torch.device(device)
        global_ranks = (
            self._global_ranks
            if self._global_ranks is not None
            else list(range(dist.get_world_size()))
        )
        world_size = len(global_ranks)

        if not self._configs:
            raise ValueError("No modules registered with parallelize().")

        pipelined, dp_size, module_ranks = self._assign_ranks(
            global_ranks, world_size
        )

        for cfg in self._configs:
            if cfg.data_parallel_size != dp_size:
                raise ValueError(
                    f"ParallelConfig.data_parallel_size={cfg.data_parallel_size} "
                    f"does not match the computed dp_size={dp_size}."
                )

        meshes: dict[int, ModalProcessGroupMesh] = {}
        layouts: dict[int, MeshLayout] = {}

        for mod_idx, (module, config) in enumerate(
            zip(self._modules, self._configs)
        ):
            modality_ranks = module_ranks[mod_idx]

            # Record every modality's rank layout on every rank (pure
            # arithmetic, no collectives) so the cross-mesh schedule can pair
            # producer and consumer ranks for meshes this rank is not part of.
            layouts[id(module)] = MeshLayout(
                global_ranks=tuple(modality_ranks),
                dp_size=dp_size,
                num_pp_stages=config.num_pp_stages,
                cp_size=config.context_parallel_size,
                tp_size=config.tensor_parallel_size,
                ep_size=config.expert_parallel_size,
            )

            # Build the mesh on EVERY rank, not just members: a ``DeviceMesh``
            # calls ``new_group`` (a world collective), so all ranks must
            # construct every modality's mesh in the same order or process-group
            # creation deadlocks. Non-members build it as a no-op and discard it;
            # only members keep and use it.
            mesh = ModalProcessGroupMesh(
                device_type=device.type,
                global_ranks=modality_ranks,
                dp_size=dp_size,
                cp_size=config.context_parallel_size,
                tp_size=config.tensor_parallel_size,
                num_pp_stages=config.num_pp_stages,
                ep_size=config.expert_parallel_size,
            )
            if not mesh.is_member:
                continue
            meshes[id(module)] = mesh

            # The model-side parallelisms (TP/CP/PP/EP) walk a Cornstarch model's
            # `_section_names()`/`hf_config`. For a modality encoder that lives on
            # its inner `.encoder`; the projector stays replicated (no registered
            # TP family) and CP/PP/EP do not apply to it. `materialize` is still
            # called on the modality encoder so the projector follows along.
            apply_target = (
                module.encoder
                if isinstance(module, CornstarchModalityEncoder)
                else module
            )

            if config.tensor_parallel_size > 1:
                apply_tensor_parallel(apply_target, mesh.tp_mesh)
            if config.context_parallel_size > 1:
                apply_context_parallel(apply_target, mesh.cp_group)
            if config.num_pp_stages > 1:
                apply_pipeline_parallel(apply_target, mesh)

            module.materialize(device, dtype=dtype)

            if config.expert_parallel_size > 1:
                apply_expert_parallel(apply_target, mesh.ep_group)

        dp_size_final, dp_rank, dp_group, grad_sync = self._build_dp_handles(
            meshes
        )

        return ParallelContext(
            modules=self._modules,
            configs=self._configs,
            meshes=meshes,
            layouts=layouts,
            dp_size=dp_size_final,
            dp_rank=dp_rank,
            dp_group=dp_group,
            gradient_synchronizer=grad_sync,
            uses_pipeline_parallel=pipelined,
        )

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

        grad_sync = GradientSynchronizer(dp_group)
        for module in self._modules:
            if id(module) in meshes:
                grad_sync.register(module)
        return dp_size, dp_rank, dp_group, grad_sync
