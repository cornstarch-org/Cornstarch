"""
Code is adopted from ColossalAI HybridParallelModule
"""

from functools import partial

import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelModule,
    _convert_floating_point,
)
from colossalai.interface import ModelWrapper
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy
from cornstarch.shardformer.shard.shardformer import ShardFormer


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
        self.stage_manager = shard_config.pipeline_stage_manager
        self.shard_config = shard_config
        self.dp_group = None
        self.tp_group = tp_group
        self.use_dpp = False
        self.require_grad_sync = True

        shardformer = ShardFormer(shard_config)
        if custom_policy is not None:
            assert isinstance(custom_policy, object)
        module, self.shared_params = shardformer.optimize(module, custom_policy)

        # setting process groups for shared parameters
        self.shared_param_process_groups = []
        for shared_param in self.shared_params:
            if len(shared_param) > 0:
                self.shared_param_process_groups.append(
                    self.stage_manager.init_process_group_by_stages(
                        list(shared_param.keys())
                    )
                )

        # setting mixed_precision
        self.mixed_precision = None
        if precision == "fp16":
            self.mixed_precision = torch.float16
        elif precision == "bf16":
            self.mixed_precision = torch.bfloat16
        if self.mixed_precision is not None:
            module = module.to(self.mixed_precision)
        module = module.to(get_accelerator().get_current_device())

        # setting input type cast when using mixed precision
        self.convert_fn = None
        if self.mixed_precision is not None:
            self.convert_fn = partial(
                _convert_floating_point, dtype=self.mixed_precision
            )

        ModelWrapper.__init__(self, module)

        self.dp_groups = dp_groups

    def sync_shared_params(self):
        for shared_param, group in zip(
            self.shared_params, self.shared_param_process_groups
        ):
            if self.stage_manager.stage in shared_param:
                param = shared_param[self.stage_manager.stage]
                dist.all_reduce(param.grad, group=group)
            # Remove barrier as it may slow down some heterogeneous pipelines
            # dist.barrier()

    def sync_sp_grads(self, grads: list[torch.Tensor] | None = None):
        # if there is no tp used, tp_group is None.
        if self.tp_group is not None:
            super().sync_sp_grads(grads)

    def sync_dp_grads(self):
        r"""
        Synchronize gradients across data parallelism (DP) if the DP group size is greater than 1.
        This function performs an all-reduce operation to combine gradients from different devices in the DP group.

        Args:
            None

        Returns:
            None
        """

        if self.dp_groups is None:
            return

        for module_name, dp_group in reversed(self.dp_groups.items()):
            module = self.module.get_submodule(module_name)
            # TODO (insujang): flatten parameters
            for param in module.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=dp_group)
                    param.grad.div_(dp_group.size())
