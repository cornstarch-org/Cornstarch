"""
Code is adopted from ColossalAI HybridParallelModule
"""

import torch
import torch.distributed as dist

from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy


class HeterogeneousParallelModule(HybridParallelModule):
    def __init__(
        self,
        module: torch.nn.Module,
        dp_groups: dict[str, dist.ProcessGroup],
        tp_group: dist.ProcessGroup,
        precision: str,
        shard_config: ShardConfig,
        custom_policy: Policy,
    ):
        super().__init__(
            module=module,
            precision=precision,
            shard_config=shard_config,
            dp_group=None,
            tp_group=tp_group,
            use_ddp=False,
            ddp_config=None,
            custom_policy=custom_policy,
        )

        self.dp_groups = dp_groups

    def sync_shared_params(self):
        super().sync_shared_params()

    def sync_dp_grads(self):
        r"""
        Synchronize gradients across data parallelism (DP) if the DP group size is greater than 1.
        This function performs an all-reduce operation to combine gradients from different devices in the DP group.

        Args:
            None

        Returns:
            None
        """

        for module_name, dp_group in self.dp_groups.items():
            if dp_group.size() == 1:
                continue

            module = self.module.get_submodule(module_name)
            # TODO (insujang): flatten parameters
            for param in module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=dp_group)
                    param.grad.div_(dp_group.size())
