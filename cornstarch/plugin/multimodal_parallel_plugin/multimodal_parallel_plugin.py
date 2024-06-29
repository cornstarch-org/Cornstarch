from typing import Any, Callable, Tuple

import numpy as np
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelPlugin
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.interface import AMPModelMixin, ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule, PipelineSchedule
from colossalai.shardformer import ShardConfig
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader

from cornstarch.models.multimodal_language_model import MultimodalModel
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_parallel_plugin import (
    ModalParallelPlugin,
)
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)


class MultimodalParallelModule(ModelWrapper, AMPModelMixin):
    pass


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

        if microbatch_size is None or num_microbatches is None:
            raise ValueError(
                "Both microbatch_size and num_microbatches must be provided."
            )
        self.microbatch_size = microbatch_size
        self.num_microbatches = num_microbatches
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

    def init_distributed(self):
        modal_templates: dict[PipelineTemplate, int] = {}
        execution_order: list[tuple[PipelineTemplate, PipelineTemplate]] = []
        for plugin in self.encoder_plugins.values():
            modal_templates[plugin.pipeline_template] = plugin.tp_size
            execution_order.append(
                (plugin.pipeline_template, self.language_model_plugin.pipeline_template)
            )
        modal_templates[self.language_model_plugin.pipeline_template] = (
            self.language_model_plugin.tp_size
        )

        # TODO: add decoders when we support multimodal generation

        self.pg_mesh = MultiModalProcessGroupMesh(modal_templates, execution_order)
        self.stage_manager = MultiModalPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.pp_groups = self.pg_mesh.get_group_along_axis(self.pg_mesh.pp_axis)

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = dist.get_world_size(group=self.pp_groups[0])

        # TODO: implement a new one if needed!
        self.schedule = OneForwardOneBackwardSchedule(
            self.stage_manager, self.num_microbatches, self.microbatch_size
        )

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager
        self.shard_config.enable_tensor_parallelism = (
            dist.get_world_size(self.tp_group) > 1
        )
        self.shard_config.__post_init__()

    def configure(
        self,
        model: MultimodalModel,
        optimizer: Optimizer | None = None,
        criterion: Callable[..., Any] | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> Tuple[Module, OptimizerWrapper, Callable[..., Any], DataLoader, LRScheduler]:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        # def get_ranks_along_pp_axis(
        #     target_modal: PipelineTemplate,
        #     stage_manager: MultiModalPipelineStageManager,
        # ) -> list:
        #     pg_mesh = stage_manager.pg_mesh
        #     stage_indices = [
        #         index
        #         for index, modal in enumerate(stage_manager.stage_index_to_modal)
        #         if modal == target_modal
        #     ]
        #     return np.take(pg_mesh.mesh, stage_indices, axis=pg_mesh.pp_axis)

        if not isinstance(model, ModelWrapper):
            for modal_name, encoder in self.encoder_plugins.items():
                module = model.get_submodule(f"{modal_name}_encoder")
                module = encoder.configure(
                    module, self.shard_config, self.stage_manager
                )
                model.add_module(modal_name, module)

            module = model.get_submodule("language_model")
            module = self.language_model_plugin.configure(module)
            model.add_module("language_model", module)

            model = MultimodalParallelModule(model)

        return model, optimizer, criterion, dataloader, lr_scheduler
