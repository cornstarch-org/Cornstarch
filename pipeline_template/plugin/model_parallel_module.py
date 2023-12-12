"""
Code is adopted from ColossalAI HybridParallelModule, but without DP.
"""

from colossalai.interface import ModelWrapper
from contextlib import contextmanager
import torch
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule

# from colossalai.interface import ModelWrapper
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy


class ModelParallelModule(HybridParallelModule):
    """A parallel module that includes model parallelism, without data parallelism.

    Due to heterogeneity in parallel modules, gradient synchronization and data parallelism
    is handled in `HeterogeneousParallelPlugin` instead of `ModelParallelModule`.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        precision: str,
        shard_config: ShardConfig,
        custom_policy: Policy,
    ):
        super().__init__(
            module=module,
            precision=precision,
            shard_config=shard_config,
            dp_group=None,
            use_ddp=False,
            ddp_config=None,
            custom_policy=custom_policy,
        )

        self.require_grad_sync = False

    def sync_grads(self):
        # sync grad across data parallel
        raise NotImplementedError(
            "sync_grads has been removed from ModelParallelModule."
        )
