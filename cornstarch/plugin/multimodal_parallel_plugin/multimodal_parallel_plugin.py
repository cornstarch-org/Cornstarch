from typing import Any, Callable, Tuple

from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.interface import OptimizerWrapper
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from cornstarch.models.multimodal_language_model import MultimodalModel
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin


class MultimodalParallelPlugin(HybridParallelPlugin):
    def __init__(
        self,
        modal_plugins: dict[str, ModalParallelPlugin],
    ):
        PipelinePluginBase.__init__(self)
        self.modal_plugins = modal_plugins

    def add_modal_plugin(self, modal_name: str, plugin: ModalParallelPlugin):
        self.modal_plugins[modal_name] = plugin

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
