"""Distributed training for Cornstarch models.

This package adds parallelism to a ``CornstarchModelBase`` without wrapping it
in ``DistributedDataParallel`` / FSDP.  The model stays a plain ``nn.Module``;
each strategy either records sharding specs on the meta module before
materialization, injects a custom forward, or runs an explicit collective in
the training loop.  Strategies compose, and the gradient sync is a manual
all-reduce so it coexists with DTensor (TP) and the custom CP/EP forwards.

Two layers
----------
**Layer 1 — composable functional primitives (foundation).**  Power users and
tests call these directly; everything else is built on them.

- **Process group mesh** (``ModalProcessGroupMesh``): a per-modality 5-D
  ``DeviceMesh`` over ``(dp, pp, cp, tp, ep)`` built in one collective call.
  Expert parallelism is a first-class mesh axis that composes with DP/CP/TP/PP.
- **Tensor parallelism** (``apply_tensor_parallel``): per-layer DTensor
  column/row sharding from a model-family plan, recorded on meta parameters
  before materialization.  Raises ``TensorParallelNotSupportedError`` for an
  unregistered model family (no silent no-op).  DTensor is used for TP only.
- **Context parallelism** (``apply_context_parallel`` + ``splitters``):
  all-gather flash attention plus data-side sequence splitters.
- **Pipeline parallelism** (``apply_pipeline_parallel`` + the
  ``TrainingSchedule`` hierarchy + P2P): explicit Megatron/ColossalAI-style
  1F1B scheduling, not DTensor-based.
- **Expert parallelism** (``apply_expert_parallel``): shards a MoE layer's
  batched experts across the EP mesh axis and routes tokens via all-to-all.
- **Data parallelism** (``GradientSynchronizer``): bucketed gradient all-reduce
  called explicitly after ``backward()``; skips expert-parallel parameters.

**Layer 2 — declarative per-modality plan (surface).**  Most users only touch
this: describe each modality with a ``ParallelConfig``, register it on a
``ParallelizationPlan``, and call ``.distribute()`` to get a ``ParallelContext``
that folds DP/CP into ``prepare_dataloader``, builds the schedule, and exposes
``sync_gradients``.
"""
from cornstarch.distributed.context_parallel import apply_context_parallel
from cornstarch.distributed.context_parallel.splitters import (
    ContextParallelSplitter,
    MakespanMinContextParallelSplitter,
    UniformContextParallelSplitter,
    ZigzagContextParallelSplitter,
)
from cornstarch.distributed.data_parallel import (
    GradientSynchronizer,
    allreduce_gradients,
)
from cornstarch.distributed.expert_parallel import apply_expert_parallel
from cornstarch.distributed.parallel_config import ParallelConfig
from cornstarch.distributed.parallelization import (
    ParallelContext,
    ParallelizationPlan,
)
from cornstarch.distributed.pipeline_parallel import apply_pipeline_parallel
from cornstarch.distributed.pipeline_parallel.schedule import (
    NonPipelineParallelSchedule,
    OneForwardOneBackwardSchedule,
    TrainingSchedule,
)
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.tensor_parallel import (
    TensorParallelNotSupportedError,
    apply_tensor_parallel,
)

__all__ = [
    # process group mesh
    "ModalProcessGroupMesh",
    # tensor parallel
    "apply_tensor_parallel",
    "TensorParallelNotSupportedError",
    # context parallel
    "apply_context_parallel",
    "ContextParallelSplitter",
    "UniformContextParallelSplitter",
    "ZigzagContextParallelSplitter",
    "MakespanMinContextParallelSplitter",
    # pipeline parallel
    "apply_pipeline_parallel",
    "TrainingSchedule",
    "NonPipelineParallelSchedule",
    "OneForwardOneBackwardSchedule",
    # expert parallel
    "apply_expert_parallel",
    # data parallel
    "GradientSynchronizer",
    "allreduce_gradients",
    # Option C surface
    "ParallelConfig",
    "ParallelizationPlan",
    "ParallelContext",
]
