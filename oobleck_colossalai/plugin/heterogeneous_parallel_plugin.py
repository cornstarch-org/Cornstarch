from types import MethodType
from typing import Callable

import torch
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    DP_AXIS,
    PP_AXIS,
    TP_AXIS,
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
)
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.checkpoint_io import CheckpointIO, HybridParallelCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule, PipelineSchedule
from colossalai.shardformer import ShardConfig
from torch.utils.data import Dataset

from oobleck_colossalai.pipeline_template import PipelineTemplate
from oobleck_colossalai.plugin.heterogeneous_dataloader import HeterogeneousDataLoader
from oobleck_colossalai.plugin.heterogeneous_parallel_module import (
    HeterogeneousParallelModule,
)
from oobleck_colossalai.process_group_mesh import HeterogeneousProcessGroupMesh
from oobleck_colossalai.shardformer.policies.auto_policy import get_autopolicy
from oobleck_colossalai.stage_manager import HeterogeneousPipelineStageManager


class HeterogeneousParallelPlugin(HybridParallelPlugin):
    """Plugin for heterogeneous parallel training.
    Tensor parallel, ZeRO, pipeline parallelism, and data parallel are combined in this plugin.
    The size of tp (tp_size) and pp should be passed in by user.
    ZeRO/TP requires a lot of communication, thus should only be done within each node.
    The size of dp is determined by the number of nodes in the given set of pipeline templates.

    In pipeline template, torch.distributed should not be initialized when the plugin is created.
    Plugin only holds meta information and later used to instantiate configuration and
    distributed intialization is deferred until `configure()` is called.

    Args:
        tp_size (int): The number of ranks for tensor parallelism.
        pipeline_templates (dict[PipelineTemplate, int]): A dictionary of pipeline templates
            and the number of pipelines to be instantiated from each template.
    """

    def __init__(
        self,
        pipelines: list[PipelineTemplate],
        tp_size: int,
        microbatch_size: int,
        num_microbatches: dict[PipelineTemplate, int],
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
    ):
        super(PipelinePluginBase).__init__()

        assert dist.is_initialized(), "torch.distributed is not initialized."

        assert (pipelines and num_microbatches) or (
            not pipelines and not num_microbatches
        ), (
            "pipelines and num_microbatches must be specified together "
            "or not specified together."
        )

        assert all(
            pipeline in num_microbatches.keys() for pipeline in pipelines
        ), "All pipelines must have a corresponding number of microbatches."

        if pipelines is None or num_microbatches is None:
            raise NotImplementedError("Auto pipeline parallelism is not supported yet.")

        num_ranks = sum(pipeline.num_stages for pipeline in pipelines) * tp_size
        assert dist.get_world_size() == num_ranks, (
            f"Number of ranks in pipeline templates ({num_ranks}) does not match "
            f"world size ({dist.get_world_size()})."
        )

        self.pipelines = pipelines
        self.tp_size = tp_size
        self.precision = precision
        self.zero_stage = 0
        self.stage_manager: HeterogeneousPipelineStageManager = None
        self.pg_mesh: HeterogeneousProcessGroupMesh = None
        self.schedule: PipelineSchedule = None
        self.global_batch_size = microbatch_size * sum(
            num_microbatches[pipeline] for pipeline in pipelines
        )
        self.microbatch_size = microbatch_size
        self.num_microbatches = num_microbatches
        self.dp_size = len(pipelines)

        self.pg_mesh = HeterogeneousProcessGroupMesh(self.pipelines, tp_size)
        self.stage_manager = HeterogeneousPipelineStageManager(self.pg_mesh, PP_AXIS)
        self.dp_groups = self.pg_mesh.get_group_along_axis(DP_AXIS)
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS)
        self.pp_group = self.pg_mesh.get_group_along_axis(PP_AXIS)

        self._pipeline_index = self.pg_mesh.coords[0][DP_AXIS]
        self.schedule = OneForwardOneBackwardSchedule(
            stage_manager=self.stage_manager,
            microbatch_size=self.microbatch_size,
        )

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=tp_size > 1,
            enable_all_optimization=enable_all_optimization,
            enable_fused_normalization=enable_fused_normalization,
            enable_flash_attention=enable_flash_attention,
            enable_jit_fused=enable_jit_fused,
            enable_sequence_parallelism=False,
            enable_sequence_overlap=False,
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

        self.ddp_config = None
        self.zero_config = None

        self.max_norm = max_norm

    def __del__(self):
        if self.pg_mesh:
            self.pg_mesh.destroy_mesh_process_groups()

    @property
    def train_batch_size(self) -> int:
        assert (
            self.stage_manager is not None
        ), "Must call set_pipeline_templates() first to determine batch size."

        return self.microbatch_size * self.num_microbatches[self._pipeline_index]

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return True

    def supported_devices(self) -> list[str]:
        return ["cuda"]

    def control_device(self) -> bool:
        return True

    def control_precision(self) -> bool:
        return True

    def support_no_sync(self) -> bool:
        return False

    def control_checkpoint_io(self) -> bool:
        return True

    def configure(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: Callable | None = None,
        dataloader: torch.utils.data.DataLoader | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> tuple[
        ModelWrapper,
        OptimizerWrapper,
        callable,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ]:
        """Instantiate pipeline templates and initialize distributed process groups."""
        if not isinstance(model, ModelWrapper):
            template = self.pipelines[self._pipeline_index]
            module_names = template.modules_per_stage[self.stage_manager.stage]

            assert (
                isinstance(self.dp_groups, list)
                and len(self.dp_groups) == len(module_names)
            ), f"Number of dp groups ({len(self.dp_groups)}) does not match the number of modules in the stage ({len(module_names)})."

            policy = get_autopolicy(template.model_name)
            policy.set_model(model)
            policy.set_pipeline_template(template)
            policy.set_shard_config(self.shard_config)

            model = HeterogeneousParallelModule(
                module=model,
                dp_groups={
                    module_name: dp_group
                    for module_name, dp_group in zip(module_names, self.dp_groups)
                }
                if self.dp_groups
                else None,
                tp_group=self.tp_group,
                precision=self.precision,
                shard_config=self.shard_config,
                custom_policy=policy,
            )

        if dataloader is None or not isinstance(dataloader, HeterogeneousDataLoader):
            raise RuntimeError(
                "dataloader must be an instance of HeterogeneousDataLoader."
            )

        # Convert num_microbatches into a flat list
        num_microbatches = [
            self.num_microbatches[pipeline] for pipeline in self.pipelines
        ]
        dataloader.configure(self._pipeline_index, num_microbatches)

        param_info = get_param_info(optimizer)
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            if self.precision in ["fp16", "bf16"]:
                optimizer = HybridParallelAMPOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    param_info=param_info,
                    precision=self.precision,
                    max_norm=self.max_norm,
                    pp_process_group=self.pp_group,
                    tp_process_group=self.tp_group,
                    **self.amp_config,
                )
            else:
                optimizer = HybridParallelNaiveOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    param_info=param_info,
                    max_norm=self.max_norm,
                    pp_process_group=self.pp_group,
                    tp_process_group=self.tp_group,
                )

        # inject update_master_params
        model.update_master_params = MethodType(optimizer.update_master_params, model)
        return model, optimizer, criterion, dataloader, lr_scheduler

    def prepare_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> HeterogeneousDataLoader:
        r"""
        Do the first-stage initialization of HeterogeneousDataLoader.
        It must finish second-stage initialization via ``configure()`` before being used for training.

        Args:
            dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random worker seed for sampling, defaults to 1024.
            drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
                is not divisible by the batch size. If False and the size of dataset is not divisible by
                the batch size, then the last batch will be smaller, defaults to False.
            pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
            num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
            kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                    `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

        Returns:
            :class:`oobleck_colossalai.plugin.heterogeneous_dataloader.HeterogeneousDataLoader`:
                A DataLoader used for training or testing.
        """
        _kwargs = kwargs.copy()
        _kwargs.pop("sampler", None)
        _kwargs.pop("batch_sampler", None)

        return HeterogeneousDataLoader(
            dataset,
            global_batch_size=self.global_batch_size,
            microbatch_size=self.microbatch_size,
            shuffle=shuffle,
            seed=seed,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> CheckpointIO:
        return HybridParallelCheckpointIO(
            self.dp_groups[0], self.pp_group, self.tp_group, self.zero_stage
        )
