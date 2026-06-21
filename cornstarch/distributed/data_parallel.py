"""Data-parallel gradient synchronization via bucketed all-reduce.

Instead of wrapping modules in ``DistributedDataParallel`` or FSDP2,
Cornstarch keeps the model unwrapped and calls ``allreduce_gradients``
explicitly after ``loss.backward()``.  This avoids DDP's parameter-
flattening requirement (which conflicts with DTensor from TP) and
FSDP2's NCCL-only constraint, while keeping the gradient-sync logic
simple and composable with all other parallelism dimensions.

Gradients are packed into flat buckets and all-reduced in bulk to
amortize collective-call latency.  DTensor gradients (from tensor
parallelism) are handled by extracting ``_local_tensor`` before
packing so the all-reduce operates on plain tensors that don't carry
DeviceMesh metadata.

The caller is responsible for using ``DistributedSampler`` (or
equivalent) so that each DP rank processes a different data shard.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

_DEFAULT_BUCKET_SIZE_MB = 25


class _GradientBucket:
    """Collects gradients into a flat buffer for a single all-reduce call.

    Gradients are copied into a contiguous buffer, all-reduced as one
    operation, then copied back to the original gradient tensors.  This
    avoids the per-parameter collective overhead that dominates when a
    model has hundreds of small parameters.

    DTensor gradients are handled transparently: the bucket operates on
    ``grad._local_tensor`` (the rank-local shard) rather than the DTensor
    wrapper, avoiding DeviceMesh conflicts with the DP process group.
    """

    def __init__(self, bucket_size_bytes: int) -> None:
        self._capacity = bucket_size_bytes
        self._grads: list[torch.Tensor] = []
        self._size = 0

    def try_add(self, grad: torch.Tensor) -> bool:
        """Add ``grad`` if it fits.  Returns False if the bucket is full."""
        grad_bytes = grad.numel() * grad.element_size()
        if self._grads and self._size + grad_bytes > self._capacity:
            return False
        self._grads.append(grad)
        self._size += grad_bytes
        return True

    @property
    def empty(self) -> bool:
        return len(self._grads) == 0

    def allreduce(self, dp_group: dist.ProcessGroup, dp_size: int) -> None:
        """Flatten, all-reduce, average, and scatter back to original grads."""
        if not self._grads:
            return

        if len(self._grads) == 1:
            dist.all_reduce(self._grads[0], op=dist.ReduceOp.SUM, group=dp_group)
            self._grads[0].div_(dp_size)
            return

        # Pack all grads into a contiguous flat buffer.
        device = self._grads[0].device
        dtype = self._grads[0].dtype
        total_numel = sum(g.numel() for g in self._grads)
        flat = torch.empty(total_numel, dtype=dtype, device=device)

        offset = 0
        for g in self._grads:
            n = g.numel()
            flat[offset : offset + n].copy_(g.reshape(-1))
            offset += n

        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=dp_group)
        flat.div_(dp_size)

        # Scatter averaged values back to original gradient tensors.
        offset = 0
        for g in self._grads:
            n = g.numel()
            g.copy_(flat[offset : offset + n].reshape(g.shape))
            offset += n


def allreduce_gradients(
    model: nn.Module,
    dp_group: dist.ProcessGroup,
    bucket_size_mb: float = _DEFAULT_BUCKET_SIZE_MB,
) -> None:
    """Average all parameter gradients across DP ranks using bucketed all-reduce.

    Gradients are packed into flat buckets of approximately
    ``bucket_size_mb`` megabytes each.  Each bucket is all-reduced in a
    single collective call, amortizing per-call latency across many
    parameters.  A final partial bucket handles any remaining gradients.

    Call this after ``loss.backward()`` and before ``optimizer.step()``.
    Parameters without gradients are silently skipped.  Expert-parallel
    parameters (tagged ``_is_expert_parallel`` by ``apply_expert_parallel``)
    are also skipped: each expert lives on a single rank and its gradient must
    not be averaged with the different experts held by other ranks.
    """
    dp_size = dist.get_world_size(dp_group)
    if dp_size <= 1:
        return

    bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
    bucket = _GradientBucket(bucket_size_bytes)

    for param in model.parameters():
        if param.grad is None:
            continue
        if getattr(param, "_is_expert_parallel", False):
            continue
        grad = param.grad._local_tensor if hasattr(param.grad, "_local_tensor") else param.grad

        if not bucket.try_add(grad):
            bucket.allreduce(dp_group, dp_size)
            bucket = _GradientBucket(bucket_size_bytes)
            bucket.try_add(grad)

    if not bucket.empty:
        bucket.allreduce(dp_group, dp_size)


class GradientSynchronizer:
    """Bucketed all-reduce gradient sync across data-parallel ranks.

    Manages gradient synchronization for one or more modules that share
    the same DP process group.  Modules are registered once after model
    construction; ``sync()`` is then called after every ``loss.backward()``
    and before ``optimizer.step()`` to average gradients across DP ranks.

    DTensor parameters from tensor parallelism are handled transparently
    by extracting the rank-local tensor before packing into buckets.

    Usage::

        grad_sync = GradientSynchronizer(mesh.dp_group)
        grad_sync.register(language_model)
        grad_sync.register(vision_module)

        # In the training loop:
        loss.backward()
        grad_sync.sync()
        optimizer.step()
    """

    def __init__(
        self,
        dp_group: dist.ProcessGroup,
        bucket_size_mb: float = _DEFAULT_BUCKET_SIZE_MB,
    ) -> None:
        self._dp_group = dp_group
        self._bucket_size_mb = bucket_size_mb
        self._modules: list[nn.Module] = []

    def register(self, module: nn.Module) -> None:
        """Register a module whose gradients will be synchronized on ``sync``."""
        self._modules.append(module)

    def sync(self) -> None:
        """All-reduce and average gradients for all registered modules."""
        for module in self._modules:
            allreduce_gradients(module, self._dp_group, self._bucket_size_mb)
