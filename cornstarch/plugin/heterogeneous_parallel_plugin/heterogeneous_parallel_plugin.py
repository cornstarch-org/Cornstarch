import warnings
from types import MethodType
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
)
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.checkpoint_io import CheckpointIO, HybridParallelCheckpointIO
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule
from colossalai.shardformer import ShardConfig
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.heterogeneous_parallel_plugin import (
    HeterogeneousDataLoader,
    HeterogeneousParallelModule,
)
from cornstarch.plugin.heterogeneous_parallel_plugin.heterogeneous_process_group_mesh import (
    HeterogeneousProcessGroupMesh,
)
from cornstarch.plugin.heterogeneous_parallel_plugin.heterogeneous_stage_manager import (
    HeterogeneousPipelineStageManager,
)
from cornstarch.shardformer.policies.auto_policy import get_autopolicy


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
        tp_size: int,
        microbatch_size: int,
        precision: str = "fp16",
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

        self.precision = precision
        self.zero_stage = 0
        self.microbatch_size = microbatch_size
        self.max_norm = max_norm
        self.tp_size = tp_size

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=None,
            pipeline_stage_manager=None,
            enable_tensor_parallelism=False,
            enable_all_optimization=False,
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

    def __del__(self):
        if hasattr(self, "pg_mesh") and self.pg_mesh:
            self.pg_mesh.destroy_mesh_process_groups()

    @property
    def train_batch_size(self) -> int:
        assert (
            self.stage_manager is not None
        ), "Must call set_pipeline_templates() first to determine batch size."

        return self.microbatch_size * self.num_microbatches[self._pipeline_index]

    @property
    def enable_pipeline_parallelism(self) -> bool:
        return self.pp_size > 1

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

    def set_pipelines(
        self,
        pipelines: list[PipelineTemplate],
        num_microbatches: dict[PipelineTemplate, int],
    ):
        assert dist.is_initialized(), "torch.distributed is not initialized."

        assert (
            pipelines and num_microbatches
        ), "pipelines and num_microbatches must be specified together."

        assert all(
            pipeline in num_microbatches.keys() for pipeline in pipelines
        ), "All pipelines must have a corresponding number of microbatches."

        num_ranks = sum(pipeline.num_stages for pipeline in pipelines) * self.tp_size
        assert dist.get_world_size() == num_ranks, (
            f"Number of ranks in pipeline templates ({num_ranks}) does not match "
            f"world size ({dist.get_world_size()})."
        )
        self.pipelines = pipelines
        self.num_microbatches = num_microbatches

        self.global_batch_size = self.microbatch_size * sum(
            num_microbatches[pipeline] for pipeline in pipelines
        )

        self.dp_axis, self.pp_axis, self.tp_axis = 0, 1, 2
        self.pg_mesh = HeterogeneousProcessGroupMesh(self.pipelines, self.tp_size)
        self._pipeline_index = self.pg_mesh.coords[0][self.dp_axis]
        self.stage_manager = HeterogeneousPipelineStageManager(
            self.pg_mesh,
            self.pp_axis,
            self.pipelines[self._pipeline_index].get_num_layers_per_stage(),
        )
        self.dp_groups = self.pg_mesh.get_group_along_axis(self.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.tp_axis)
        self.pp_group = self.pg_mesh.get_group_along_axis(self.pp_axis)

        self.dp_size = len(pipelines)
        self.pp_size = dist.get_world_size(self.pp_group)

        self.schedule = OneForwardOneBackwardSchedule(
            stage_manager=self.stage_manager,
            microbatch_size=self.microbatch_size,
        )

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager
        self.shard_config.enable_tensor_parallelism = self.tp_size > 1
        self.shard_config.__post_init__()

    def configure(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer | None = None,
        criterion: Callable | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
        forced: bool = False,
    ) -> tuple[ModelWrapper, OptimizerWrapper, callable, DataLoader, LRScheduler]:
        """Instantiate pipeline templates and initialize distributed process groups."""

        if forced or not isinstance(model, ModelWrapper):
            my_pipeline = self.pipelines[self._pipeline_index]
            module_names = my_pipeline.modules_per_stage[self.stage_manager.stage]

            assert isinstance(self.dp_groups, list) and len(self.dp_groups) == len(
                module_names
            ), f"Number of dp groups ({len(self.dp_groups)}) does not match the number of modules in the stage ({len(module_names)})."

            module = model.module if isinstance(model, ModelWrapper) else model

            policy = get_autopolicy(my_pipeline.model_name)
            policy.set_model(module)
            policy.set_shard_config(self.shard_config)

            model = HeterogeneousParallelModule(
                module=module,
                dp_groups=(
                    {
                        module_name: dp_group
                        for module_name, dp_group in zip(module_names, self.dp_groups)
                    }
                    if self.dp_groups
                    else None
                ),
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

        if optimizer is not None:
            if not isinstance(optimizer, OptimizerWrapper):
                param_info = get_param_info(optimizer)
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
            elif forced and isinstance(optimizer, OptimizerWrapper):
                optimizer.model = model
                optimizer.stage_manager = self.stage_manager
                optimizer.shared_params = model.shared_params
                optimizer.tp_pg = self.tp_group
                optimizer.pp_pg = self.pp_group
                optimizer.tp_size = self.tp_size
                optimizer.pp_size = self.pp_size

            # inject update_master_params
            model.update_master_params = MethodType(
                optimizer.update_master_params, model
            )

        return model, optimizer, criterion, dataloader, lr_scheduler

    def execute_pipeline(
        self,
        data_iter: Iterator,
        model: HeterogeneousParallelModule,
        criterion: Callable[[Any, Any], torch.Tensor],
        optimizer: (
            HybridParallelNaiveOptimizer | HybridParallelAMPOptimizer | None
        ) = None,
        return_loss: bool = True,
        return_outputs: bool = False,
    ) -> dict:
        # assert self.enable_pipeline_parallelism, "pipeline parallelism is not enabled"

        if return_outputs:
            warnings.warn(
                "return_outputs may lead to significant extra memory consumption."
            )

        # Create a context for gradient synchronization based on the optimizer type.
        # If it's a HybridParallelZeroOptimizer, use optimizer.no_sync(); otherwise, use model.no_sync().
        # This is to avoid redundant gradient reduction in pipeline parallelism (multiple microbatch values should be reduced once),
        # so we disable it, performing manual reduction instead.
        with model.no_sync():
            outputs = self.schedule.forward_backward_step(
                model, data_iter, criterion, optimizer, return_loss, return_outputs
            )

        # run with gradients accumulation
        if not model.require_grad_sync:
            return outputs

        # Synchronize the grads of shared parameters of the model.
        model.sync_shared_params()

        # Check if the optimizer is a HybridParallelZeroOptimizer and synchronize data parallelism gradients if so.
        # Otherwise, synchronize data parallelism gradients of the model.
        # This is because these are two different forms of data parallelism.
        model.sync_dp_grads()

        return outputs

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
            :class:`cornstarch.plugin.heterogeneous_dataloader.HeterogeneousDataLoader`:
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
