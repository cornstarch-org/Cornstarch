from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard.shard_config import ShardConfig as ColossalShardConfig

from cornstarch.pipeline_template import PipelineTemplate


@dataclass
class ShardConfig(ColossalShardConfig):
    tensor_parallel_process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel_process_group: Optional[dist.ProcessGroup] = None
    pipeline_stage_manager: Optional[PipelineStageManager] = None
    pipeline_template: Optional[PipelineTemplate] = None
    enable_tensor_parallelism: bool = True
    enable_all_optimization: bool = False
    enable_fused_normalization: bool = False
    enable_flash_attention: bool = False
    enable_jit_fused: bool = False
    parallel_output: bool = True
    make_vocab_size_divisible_by: int = 64
