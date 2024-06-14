from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type

import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelCheckpointIO
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.checkpoint_io import CheckpointIO
from colossalai.cluster import ProcessGroupMesh
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import PipelineSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer.policies.base_policy import Policy
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset


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
        pp_size: int,
        precision: str = "fp16",
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0,
        cpu_offload: bool = False,
        custom_policy: Policy = None,
        enable_metadata_cache: bool = True,
        make_vocab_size_divisible_by: int = 64,
    ):
        super().__init__()

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.precision = precision
        self.cpu_offload = cpu_offload
        self.enable_all_optimization = enable_all_optimization
        self.enable_fused_normalization = enable_fused_normalization
        self.enable_flash_attention = enable_flash_attention
        self.enable_jit_fused = enable_jit_fused
        self.custom_policy: Policy = custom_policy

        self.dp_axis, self.pp_axis, self.tp_axis = 0, 1, 2

        self.stage_manager: PipelineStageManager = None
        self.pg_mesh: ProcessGroupMesh = None
        self.schedule: Type[PipelineSchedule] = None
        self.tp_group: dist.ProcessGroup = None
        self.dp_group: dist.ProcessGroup = None
        self.pp_group: dist.ProcessGroup = None

        self.amp_config = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
        )

        self.max_norm = max_norm
        self.enable_metadata_cache = enable_metadata_cache
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return self.pp_size > 1

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
        optimizer: Optional[Optimizer] = None,
        criterion: Optional[Callable] = None,
        dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> Tuple[nn.Module, OptimizerWrapper, Callable, DataLoader, LRScheduler]:
        raise NotImplementedError

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

    def enable_lora(
        self, model: nn.Module, pretrained_dir: str, lora_config: Dict
    ) -> nn.Module:
        raise NotImplementedError

    def get_checkpoint_io(self) -> CheckpointIO:
        if self.tp_group is None or self.dp_group is None or self.pp_group is None:
            raise ValueError("all groups must not be None")
        return HybridParallelCheckpointIO(
            self.tp_group, self.dp_group, self.pp_group, zero_stage=0
        )
