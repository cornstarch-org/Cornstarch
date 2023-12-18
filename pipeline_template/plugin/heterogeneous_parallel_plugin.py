import itertools
import random
from logging import getLogger
from types import MethodType
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from colossalai.booster.plugin.hybrid_parallel_plugin import (
    DP_AXIS,
    PP_AXIS,
    TP_AXIS,
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    HybridParallelZeroOptimizer,
    get_param_info,
)
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule import OneForwardOneBackwardSchedule, PipelineSchedule
from colossalai.shardformer import ShardConfig, ShardFormer
from colossalai.shardformer.policies.auto_policy import get_autopolicy
from colossalai.shardformer.policies.base_policy import Policy
from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.plugin.heterogeneous_parallel_module import (
    HeterogeneousParallelModule,
)
from pipeline_template.process_group_mesh import HeterogeneousProcessGroupMesh
from pipeline_template.shardformer.pipeline_template_policy import (
    PipelineTemplatePolicyWrapper,
)
from pipeline_template.stage_manager import HeterogeneousPipelineStageManager


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
        precision: str = "fp16",
        zero_stage: int = 0,
        zero_size: int = 1,
        enable_all_optimization: bool = False,
        enable_fused_normalization: bool = False,
        enable_flash_attention: bool = False,
        enable_jit_fused: bool = False,
        num_microbatches: list[int] | None = None,
        microbatch_size: int | None = None,
        initial_scale: float = 2**16,
        min_scale: float = 1,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        hysteresis: int = 2,
        max_scale: float = 2**32,
        max_norm: float = 0,
        zero_bucket_size_in_m: int = 12,
        cpu_offload: bool = False,
        communication_dtype: torch.dtype | None = None,
        overlap_communication: bool = True,
    ):
        super(PipelinePluginBase).__init__()

        self.precision = precision
        self.zero_stage = zero_stage
        self.stage_manager: HeterogeneousPipelineStageManager = None
        self.pg_mesh: HeterogeneousProcessGroupMesh = None
        self.schedule: PipelineSchedule = None
        self.num_microbatches = num_microbatches
        self.microbatch_size = microbatch_size

        logger = getLogger(__name__)

        assert zero_stage in (0, 1, 2)

        self.shard_config = dict(
            tp_size=tp_size,
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

        self.zero_config = dict(
            zero_stage=zero_stage,
            zero_size=zero_size,
            reduce_bucket_size=zero_bucket_size_in_m * 1024 * 1024,
            communication_dtype=communication_dtype,
            overlap_communication=overlap_communication,
            cpu_offload=cpu_offload,
            partition_grad=(self.zero_stage == 2),
        )

        self.max_norm = max_norm

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

    def set_pipeline_templates(
        self, pipeline_templates: dict[PipelineTemplate, int] | None
    ):
        """Set pipeline templates and instantiate pipeline stage manager.

        If pipeline templates are set, thje plugin will use the given
        templates to instantiate pipeline parallel training.
        Given pipeline templates must cover all ranks.
        """
        assert dist.is_initialized(), "torch.distributed is not initialized."

        num_ranks = sum(
            template.num_gpus * num_pipelines
            for template, num_pipelines in pipeline_templates.items()
        )
        assert dist.get_world_size() == num_ranks, (
            f"Number of ranks in pipeline templates does not match "
            f"world size ({dist.get_world_size()})."
        )

        dp_size = sum(pipeline_templates.values())

        if sum(pipeline_templates.values()) > 1:
            assert self.num_microbatches is not None and dp_size == len(
                self.num_microbatches
            ), "number of pipelines should match the length of num_microbatches list."
            assert (
                self.microbatch_size is not None
            ), "microbatch_size must be specified when using pipeline parallelism"
            assert (
                self.zero_config["zero_stage"] <= 1
            ), "zero stage must be 0 or 1 when using pipeline parallelism"

        self.pipeline_templates = pipeline_templates
        self.dp_size = dp_size

        self.pg_mesh = HeterogeneousProcessGroupMesh(
            self.pipeline_templates, self.shard_config["tp_size"]
        )
        self.stage_manager = HeterogeneousPipelineStageManager(self.pg_mesh, PP_AXIS)
        self.tp_group = self.pg_mesh.get_group_along_axis(TP_AXIS)
        self.dp_group = self.pg_mesh.get_group_along_axis(DP_AXIS)
        self.pp_group = self.pg_mesh.get_group_along_axis(PP_AXIS)

        self._pipeline_index_to_pipeline: dict[int, PipelineTemplate] = {
            index: pipeline_template
            for index, pipeline_template in enumerate(
                itertools.chain.from_iterable(
                    itertools.repeat(pipeline_template, num_pipelines)
                    for pipeline_template, num_pipelines in pipeline_templates.items()
                )
            )
        }
        self._pipeline_index = self.pg_mesh.coords[0][DP_AXIS]
        self.schedule = OneForwardOneBackwardSchedule(
            self.stage_manager,
            self.num_microbatches[self._pipeline_index],
            self.microbatch_size,
        )

        self.shard_config = ShardConfig(
            tensor_parallel_process_group=self.tp_group,
            pipeline_stage_manager=self.stage_manager,
            enable_tensor_parallelism=self.shard_config["tp_size"] > 1,
            enable_all_optimization=self.shard_config["enable_all_optimization"],
            enable_fused_normalization=self.shard_config["enable_fused_normalization"],
            enable_flash_attention=self.shard_config["enable_flash_attention"],
            enable_jit_fused=self.shard_config["enable_jit_fused"],
            enable_sequence_parallelism=self.shard_config[
                "enable_sequence_parallelism"
            ],
            enable_sequence_overlap=self.shard_config["enable_sequence_overlap"],
        )

    def configure(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: Callable | None = None,
        dataloader: torch.utils.data.DataLoader | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> tuple[
        torch.nn.Module,
        OptimizerWrapper,
        callable,
        torch.utils.data.DataLoader,
        torch.optim.lr_scheduler.LRScheduler,
    ]:
        """Instantiate pipeline templates and initialize distributed process groups."""
        assert self.pipeline_templates, "Pipeline templates are not set."

        if not isinstance(model, ModelWrapper):
            dp_groups = self.pg_mesh.get_group_along_axis(DP_AXIS)
            template = self._pipeline_index_to_pipeline[self._pipeline_index]
            module_names = template.modules_per_stage[self.stage_manager.stage]
            if dp_groups:
                assert isinstance(dp_groups, list) and len(dp_groups) == len(
                    module_names
                ), f"Number of dp groups ({len(dp_groups)}) does not match the number of modules in the stage ({len(module_names)})."
            else:
                assert (
                    sum(self.pipeline_templates.values()) == 1
                ), "There are more than 1 dp_groups but dp_group is None."

            policy = get_autopolicy(model)
            policy.set_model(model)
            policy.set_shard_config(self.shard_config)
            policy = PipelineTemplatePolicyWrapper(template).wrap(policy)

            model = HeterogeneousParallelModule(
                module=model,
                dp_groups={
                    module_name: dp_group
                    for module_name, dp_group in zip(module_names, dp_groups)
                }
                if dp_groups
                else None,
                tp_group=self.tp_group,
                precision=self.precision,
                shard_config=self.shard_config,
                custom_policy=policy,
            )

        param_info = get_param_info(optimizer)
        if optimizer is not None and not isinstance(optimizer, OptimizerWrapper):
            if self.zero_stage == 0:
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
            else:
                assert (
                    self.tp_size > 1
                ), "Please use Zero when tenosr parallel size is greater than 1."
                assert (
                    self.precision != "fp32"
                ), "Please set precision to 'fp16' or 'bf16' when using ZeRO."
                optimizer = HybridParallelZeroOptimizer(
                    optimizer,
                    model,
                    use_pipeline=self.enable_pipeline_parallelism,
                    param_info=param_info,
                    dp_process_group=self.dp_group,
                    tp_process_group=self.tp_group,
                    pp_process_group=self.pp_group,
                    verbose=True,
                    clip_grad_norm=self.max_norm,
                    **self.zero_config,
                    **self.amp_config,
                )

            # inject update_master_params
            model.update_master_params = MethodType(
                optimizer.update_master_params, model
            )
            return model, optimizer, criterion, dataloader, lr_scheduler

    def prepare_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        **kwargs,
    ):
        r"""
        Prepare a dataloader for distributed training. The dataloader will be wrapped by
        `torch.utils.data.DataLoader` and `torch.utils.data.DistributedSampler`.


        Args:
            dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            seed (int, optional): Random worker seed for sampling, defaults to 1024.
            add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
            drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
                is not divisible by the batch size. If False and the size of dataset is not divisible by
                the batch size, then the last batch will be smaller, defaults to False.
            pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
            num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
            kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                    `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.

        Returns:
            :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
        """
        _kwargs = kwargs.copy()
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=self.dp_size,
            rank=self._pipeline_index,
            shuffle=shuffle,
        )

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )
