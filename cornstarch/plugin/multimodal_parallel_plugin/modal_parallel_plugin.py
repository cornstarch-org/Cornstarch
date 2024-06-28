from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelCheckpointIO
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.checkpoint_io import CheckpointIO
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import PipelineSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer import ShardConfig
from colossalai.shardformer.policies.base_policy import Policy
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.auto_policy import get_autopolicy


class ModalParallelPlugin(PipelinePluginBase):
    """
    Plugin for hybrid parallel training of submodalities of multimodal model.
    Tensor parallelism and pipeline parallelism can be combined in this plugin.

    The plugin is similar to `HybridParallelPlugin`, but designed for multimodal model training.
    It adopts hierarchical architecture, where each modality is configured with `ModalParallelPlugin`,
    and the multimodal model is configured with `MultimodalParallelPlugin`.

    Differences of `ModalParallelPlugin` from `HybridParallelPlugin`:
    - `ModalParallelPlugin` does not infer `dp_size` and `dp_size` will be calculated in `MultimodalParallelPlugin`.


    Args:
        tp_size (int): The size of tensor parallelism.
                       Tensor parallelism will not be used when tp_size is set to 1.
        pp_size (int): The number of pipeline stages in pipeline parallelism.
                       Pipeline parallelism will not be used when pp_size is set to 1.
        precision (str): The precision of the model. Defaults to 'fp16'.
                         Auto-mixied precision will be used when this argument is set to 'fp16' or 'bf16',
                         otherwise model is trained with 'fp32'.
        enable_all_optimization (bool, optional): Whether to switch on all the optimizations supported by Shardformer.
                                                  Currently all the optimization methods include fused normalization, flash attention and JIT.
                                                  Defaults to False.
        enable_fused_normalization (bool, optional): Whether to switch on fused normalization in Shardformer.
                                                     Defaults to False.
        enable_flash_attention (bool, optional): Whether to switch on flash attention in Shardformer.
                                                 Defaults to False.
        enable_jit_fused (bool, optional): Whether to switch on JIT in Shardformer.
                                           Defaults to False.
        initial_scale (float, optional): The initial loss scale of AMP. Defaults to 2**16.
        min_scale (float, optional): The minimum loss scale of AMP. Defaults to 1.
        growth_factor (float, optional): The multiplication factor for increasing loss scale when using AMP.
                                         Defaults to 2.
        backoff_factor (float, optional): The multiplication factor for decreasing loss scale when using AMP.
                                          Defaults to 0.5.
        growth_interval (int, optional): The number of steps to increase loss scale when no overflow occurs when using AMP.
                                         Defaults to 1000.
        hysteresis (int, optional):  The number of overflows before decreasing loss scale when using AMP.
                                     Defaults to 2.
        max_scale (float, optional): The maximum loss scale of AMP. Defaults to 2**32.
        max_norm (float, optional): Maximum norm for gradient clipping. Defaults to 0.
        cpu_offload (bool, optional): Whether to offloading optimizer states to CPU. Defaults to False.
        custom_policy (Policy, optional): Custom policy for Shardformer. Defaults to None.
        enable_metadata_cache (bool, optional): Whether to enable metadata cache for pipeline parallelism. Defaults to True.
        make_vocab_size_divisible_by (int, optional): it's used when padding the vocabulary size, to make it choose an faster kenel.
                                                      Default to 64.
    """

    def __init__(
        self,
        tp_size: int,
        pipeline_template: PipelineTemplate = None,
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        parallel_output: bool = True,
        cpu_offload: bool = False,
        custom_policy: Policy = None,
        make_vocab_size_divisible_by: int = 64,
    ):
        super().__init__()

        self.tp_size = tp_size
        self.pipeline_template = pipeline_template
        self.precision = precision
        self.cpu_offload = cpu_offload
        self.custom_policy = custom_policy

        if self.cpu_offload:
            raise NotImplementedError("CPU offload is not supported yet.")

        self.stage_manager: PipelineStageManager = None
        self.pg_mesh: ProcessGroupMesh = None
        self.schedule: Type[PipelineSchedule] = None
        self.tp_group: dist.ProcessGroup = None
        self.dp_group: dist.ProcessGroup = None
        self.pp_group: dist.ProcessGroup = None

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=None,
            enable_tensor_parallelism=False,
            enable_all_optimization=False,
            enable_flash_attention=enable_flash_attention,
            enable_fused_normalization=enable_fused_normalization,
            enable_jit_fused=enable_jit_fused,
            enable_sequence_parallelism=False,
            enable_sequence_overlap=False,
            sequence_parallel_process_group=None,
            parallel_output=parallel_output,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        )

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return self.pipeline_template and self.pipeline_template.num_stages > 1

    def supported_devices(self) -> list[str]:
        return ["cuda"]

    def supported_precisions(self) -> list[str]:
        return ["fp16", "bf16", "fp32"]

    def control_device(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return True

    def support_no_sync(self) -> bool:
        return True

    def support_lora(self) -> bool:
        return False

    def control_checkpoint_io(self) -> bool:
        return True

    def configure(
        self,
        model: nn.Module,
        ranks: list[list[list[int]]],
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        assert dist.is_initialized(), "torch.distributed is not initialized."

        if not isinstance(model, ModelWrapper):
            policy = get_autopolicy(self.pipeline_template.model_name)
            policy.set_model(model)
            policy.set_shard_config(self.shard_config)

    def execute_pipeline(
        self,
        data_iter: Iterator,
        model: ModelWrapper,
        criterion: Callable[[Any, Any], Tensor],
        optimizer: OptimizerWrapper | None = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> Dict:
        raise NotImplementedError

    def no_sync(self, model: nn.Module, optimizer: OptimizerWrapper) -> Iterator[None]:
        raise NotImplementedError

    def enable_lora(
        self, model: nn.Module, pretrained_dir: str, lora_config: dict
    ) -> nn.Module:
        raise NotImplementedError

    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        raise NotImplementedError

    def get_checkpoint_io(self) -> CheckpointIO:
        if self.tp_group is None or self.dp_group is None or self.pp_group is None:
            raise ValueError("all groups must not be None")
        return HybridParallelCheckpointIO(
            self.tp_group, self.dp_group, self.pp_group, zero_stage=0
        )
