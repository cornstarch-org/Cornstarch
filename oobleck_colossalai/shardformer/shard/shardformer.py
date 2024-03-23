from typing import Optional

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.shard.sharder import ModelSharder as ColossalModelSharder
from colossalai.shardformer.shard.shardformer import ShardConfig
from colossalai.shardformer.shard.shardformer import ShardFormer as ColossalShardFormer
from torch import Tensor, nn

from oobleck_colossalai.shardformer.shard.placeholder import ParameterPlaceholder


class ModelSharder(ColossalModelSharder):
    def set_tensors_to_placeholder(
        self, model: nn.Module, exclude: set[nn.Module] = set()
    ) -> None:
        """Set all parameters and buffers of model to ParameterPlaceholder instances"""
        if model in exclude:
            return

        for child in model.children():
            self.set_tensors_to_placeholder(child, exclude=exclude)

        for n, p in list(model.named_parameters(recurse=False)):
            setattr(model, n, ParameterPlaceholder(p))
        for n, buf in list(model.named_buffers(recurse=False)):
            setattr(model, n, ParameterPlaceholder(buf))

    def _release_unheld_layers(self) -> Optional[set[nn.Module]]:
        if self.shard_config and self.shard_config.pipeline_stage_manager:
            held_layers = self.policy.get_held_layers()
            self.set_tensors_to_placeholder(self.model, exclude=set(held_layers))
            return set(self._get_recursive_held_layers(held_layers))
        return None


class ShardFormer(ColossalShardFormer):
    """
    Parallelize model based on the given config and policy.

    In original ColossalAI's ShardFormer, it removes all unheld layers
    by setting their parameters and buffers to None.

    Oobleck training may restore those parameters and buffers, so we
    instead replace them with placeholders. This way, we can restore
    the original model when failures happen and model sharding is changed.
    """

    def __init__(self, shard_config: ShardConfig):
        self.shard_config = shard_config

    def optimize(
        self, model: nn.Module, policy: Policy = None
    ) -> tuple[nn.Module, list[dict[int, Tensor]]]:
        r"""
        This method will optimize the model based on the given policy.

        Args:
            model (`torch.nn.Model`): the origin huggingface model
            shard_config (`ShardConfig`): the config for distribute information
            policy (`Policy`): the custom policy for sharding

        Returns: the sharded model and the shared parameters
        """
        sharder = ModelSharder(
            model=model, shard_config=self.shard_config, policy=policy
        )
        shared_params = sharder.shard()
        return model, shared_params
