from __future__ import annotations

import torch.nn as nn

from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.shardformer.shard.shard_config import ShardConfig


class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines.

    TODO (insujang): analyze optimal partitioning of model layers to pipeline stages.
    TODO (insujang): implement to assign different number of layers and different number of
    GPUs to each stage.
    """

    def __init__(
        self,
        num_nodes: int,
        gpus_per_node: int,
        modules_per_stage: list[list[nn.Module]],
    ):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.modules_per_stage = modules_per_stage

    @property
    def num_layers(self) -> int:
        return sum(len(stage) for stage in self.modules_per_stage)

    @property
    def num_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node

    @staticmethod
    def create_pipeline_template(
        node_ids: list[str],
        gpus_per_stage: list[int],
        module_names: list[str],
    ):
        """Create a pipeline template.

        Analyzing the given modules, this method creates a pipeline template
        that distributes modules to pipeline stages evenly considering
        their computational costs.
        """
        raise NotImplementedError("This method has not been implemented yet.")
