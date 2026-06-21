"""Tensor parallelism via PyTorch DTensor.

Tensor parallelism shards each repeated layer's linear weights across the TP
ranks: attention/FFN input projections are column-parallel and output
projections are row-parallel, so a forward pass needs one all-reduce per
attention and per FFN block.  The split is described by a per-model-family
plan in ``plans.py`` (submodule path -> ``ParallelStyle``) and applied with
``torch.distributed.tensor.parallel.parallelize_module``.

``apply_tensor_parallel`` runs on the meta module before ``materialize()``:
``parallelize_module`` records the DTensor sharding spec on the meta
parameters, and real (already-sharded) storage is allocated during
materialization.

DTensor is used *only* for TP weight sharding here — it is deliberately not
extended to pipeline parallelism or FSDP-style data parallelism (those use the
explicit Megatron/ColossalAI-style schedule and manual gradient all-reduce
instead).

Unlike the earlier trial, a model family with no registered TP plan raises
``TensorParallelNotSupportedError`` rather than silently leaving the model
unsharded — a silent no-op would let a caller believe a model is tensor
parallel when it is not, producing wrong sharding math downstream.
"""
from __future__ import annotations

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

from cornstarch.distributed.tensor_parallel.plans import get_tp_plan
from cornstarch.models.model_base import CornstarchModelBase


class TensorParallelNotSupportedError(NotImplementedError):
    """Raised when ``apply_tensor_parallel`` has no plan for a model family."""


def apply_tensor_parallel(module: CornstarchModelBase, tp_mesh: DeviceMesh) -> None:
    """Wrap linear layers in each repeated layer with DTensor column/row sharding.

    Must be called before ``module.materialize()`` so the sharding spec is
    recorded on meta parameters; actual storage is allocated during
    materialization on the correct device.

    Raises
    ------
    TensorParallelNotSupportedError
        If no TP plan is registered for the module's model family.  This is an
        explicit error (not a silent no-op) so callers cannot mistake an
        unsharded model for a tensor-parallel one.
    """
    config_class_name = type(module.hf_config).__name__
    plan = get_tp_plan(config_class_name)
    if plan is None:
        raise TensorParallelNotSupportedError(
            f"No tensor-parallel plan is registered for model family "
            f"'{config_class_name}'. Register a per-layer TP plan in "
            f"cornstarch/distributed/tensor_parallel/plans.py before applying "
            f"tensor parallelism to this model."
        )

    _, layers_name, _ = module._section_names()
    for layer in getattr(module, layers_name):
        parallelize_module(layer, tp_mesh, plan)
