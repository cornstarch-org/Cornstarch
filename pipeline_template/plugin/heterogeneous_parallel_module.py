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
        precision: str,
        shard_config: ShardConfig,
        custom_policy: Policy,
    ):
        for module_name in dp_groups.keys():
            assert (
                module.get_submodule(module_name) is not None
            ), f"Submodule {module_name} is not found in module list."

        super().__init__(
            module=module,
            precision=precision,
            shard_config=shard_config,
            dp_group=None,
            use_ddp=False,
            ddp_config=None,
            custom_policy=custom_policy,
        )

        self.dp_groups = dp_groups

    def sync_shared_params(self):
        super().sync_shared_params()

    def sync_grads(self):
        # sync grad across data parallel
        if len(self.dp_groups) == 1:
            return

        for module_name, dp_group in self.dp_groups.items():
            module = self.module.get_submodule(module_name)
            for param in module.parameters():
                dist.all_reduce(param.grad.data, group=dp_group)
                param.grad.div_(dp_group.size())
