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
    MAKESPAN_MIN = "makespan_min"


@dataclass
class ShardConfig(ColossalShardConfig):
    tensor_parallel_process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel_process_group: Optional[dist.ProcessGroup] = None
    enable_sequence_parallelism: bool = False
    sequence_parallelism_mode: str = None
    enable_sequence_overlap: bool = False
    context_parallel_distribution_mode: ContextParallelDistributionMode = (
        ContextParallelDistributionMode.MAKESPAN_MIN
    )
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

        if self.sequence_parallelism_mode is not None:
            assert self.sequence_parallelism_mode in [
                "ring_attn",
                "all_to_all",
            ], f"Invalid sequence parallelism mode {self.sequence_parallelism_mode}"
