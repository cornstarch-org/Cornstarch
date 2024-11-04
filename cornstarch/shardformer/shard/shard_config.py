from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.shard.shard_config import ShardConfig as ColossalShardConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.layers.utils import SUPPORT_RING_ATTN_DISTRIBUTION_MODE


@dataclass
class ShardConfig(ColossalShardConfig):
    tensor_parallel_process_group: Optional[dist.ProcessGroup] = None
    sequence_parallel_process_group: Optional[dist.ProcessGroup] = None
    enable_sequence_parallelism: bool = False
    sequence_parallelism_mode: str = None
    ring_attention_distribution_mode: str = None
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

        if (
            self.enable_sequence_parallelism
            and self.sequence_parallelism_mode == "ring_attn"
        ):
            if self.ring_attention_distribution_mode is None:
                self.ring_attention_distribution_mode = "zigzag"

            assert (
                self.ring_attention_distribution_mode
                in SUPPORT_RING_ATTN_DISTRIBUTION_MODE
            ), f"Ring attention distribution mode {self.ring_attention_distribution_mode} is not in the supported list {SUPPORT_RING_ATTN_DISTRIBUTION_MODE}"
