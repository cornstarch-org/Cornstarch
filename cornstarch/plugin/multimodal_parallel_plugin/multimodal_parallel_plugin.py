from typing import Any, Callable, Tuple

import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.interface import OptimizerWrapper
from colossalai.shardformer import ShardConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from cornstarch.models.multimodal_language_model import MultimodalModel
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin


class MultimodalParallelPlugin(HybridParallelPlugin):
    """Plugin for multimodal language model.
    Tensor parallel, pipeline parallel, and data parallel are combined in this plugin.
    Each modal has its own parallel configuration defined in ModalParallelPlugin.
    """

    def __init__(
        self,
        encoder_plugins: dict[str, ModalParallelPlugin] = None,
        language_model_plugin: ModalParallelPlugin | None = None,
        precision: str = "fp16",
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        num_microbatches: int = None,
        microbatch_size: int = None,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0,
        parallel_output: bool = True,
        make_vocab_size_divisible_by: int = 64,
    ):
        PipelinePluginBase.__init__(self)
        self.encoder_plugins = encoder_plugins
        self.language_model_plugin = language_model_plugin

        self.precision = precision
        self.zero_stage = 0
        self.microbatch_size = microbatch_size
        self.num_microbatch = self.num_microbatch
        self.global_batch_size = microbatch_size * num_microbatches
        self.max_norm = max_norm

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            enable_tensor_parallelism=False,
            pipeline_stage_manager=None,
            enable_all_optimization=False,
            enable_fused_normalization=enable_fused_normalization,
            enable_flash_attention=enable_flash_attention,
            enable_jit_fused=enable_jit_fused,
            enable_sequence_parallelism=False,
            enable_sequence_overlap=False,
            parallel_output=parallel_output,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by,
        )

        self.amp_config = dict(
            initial_scale=initial_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            hysteresis=hysteresis,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    def add_encoder_plugins(self, name: str, plugin: ModalParallelPlugin):
        self.encoder_plugins[name] = plugin

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return True

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
        """LoRA must manually be added to each modal before generating the plugin."""
        return False

    def control_checkpoint_io(self) -> bool:
        return True

    def configure(
        self,
        model: MultimodalModel,
        optimizer: Optimizer | None = None,
        criterion: Callable[..., Any] | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> Tuple[Module, OptimizerWrapper, Callable[..., Any], DataLoader, LRScheduler]:
        raise NotImplementedError
