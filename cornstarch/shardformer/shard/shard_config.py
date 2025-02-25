from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard.shard_config import ShardConfig as ColossalShardConfig

from cornstarch.pipeline_template import PipelineTemplate


class ContextParallelDistributionMode(Enum):
    """
    Enum class for the context parallel distribution mode
    """

    UNIFORM = "uniform"
    ZIGZAG = "zigzag"
    MAKESPAN_MAIN = "makespan_main"


@dataclass
class ShardConfig(ColossalShardConfig):
    tensor_parallel_process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel_process_group: Optional[dist.ProcessGroup] = None
    enable_sequence_parallelism: bool = False
    sequence_parallelism_mode: ContextParallelDistributionMode = (
        ContextParallelDistributionMode.UNIFORM
    )
    enable_sequence_overlap: bool = False
    context_parallel_distribution_mode: str = None
    pipeline_stage_manager: Optional[PipelineStageManager] = None
    pipeline_template: Optional[PipelineTemplate] = None
    enable_tensor_parallelism: bool = True
    enable_all_optimization: bool = False
    enable_fused_normalization: bool = False
    enable_flash_attention: bool = False
    enable_jit_fused: bool = False
    parallel_output: bool = True
    make_vocab_size_divisible_by: int = 64

    def __post_init__(self):
        super().__post_init__()

        assert (
            not self.enable_sequence_parallelism
        ), "Sequence parallelism is currently not supported"
