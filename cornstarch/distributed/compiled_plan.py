"""Public compiled-partition API and compatibility installation."""

from __future__ import annotations

from cornstarch.distributed._compiled_plan_base import (
    CompiledParallelizationPlan,
    LogicalStateManifest,
    LogicalStateManifestEntry,
    ParallelContext,
    ParallelizationPlan,
    PipelineStageSpec,
)
from cornstarch.distributed.parallelization import ParallelContext as _BaseContext
from cornstarch.distributed.parallelization import ParallelizationPlan as _BasePlan


# ``cornstarch.distributed.parallelization.ParallelizationPlan`` is an existing
# documented import path. Install only the additive lifecycle methods there;
# leave its materialize() implementation untouched for strict compatibility.
for _name in (
    "compile",
    "_default_stage_specs",
    "_activate_compiled",
    "_restore_module",
    "_restore_blueprint",
):
    setattr(_BasePlan, _name, getattr(ParallelizationPlan, _name))


def _close_base_context(self: _BaseContext) -> None:
    if getattr(self, "_closed", False):
        return
    for name in (
        "_meshes",
        "_layouts",
        "_cp_gradient_synchronizers",
        "_cross_mesh_groups",
    ):
        value = getattr(self, name, None)
        clear = getattr(value, "clear", None)
        if callable(clear):
            clear()
    self._gradient_synchronizer = None
    self._dp_group = None
    self._closed = True


if not hasattr(_BaseContext, "close"):
    _BaseContext.close = _close_base_context


__all__ = [
    "CompiledParallelizationPlan",
    "LogicalStateManifest",
    "LogicalStateManifestEntry",
    "ParallelContext",
    "ParallelizationPlan",
    "PipelineStageSpec",
]
