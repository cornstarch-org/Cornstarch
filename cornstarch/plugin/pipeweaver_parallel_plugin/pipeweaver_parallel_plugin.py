from __future__ import annotations

import inspect
import random
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import MethodType
from typing import Any, Callable, Optional, Tuple

from colossalai.checkpoint_io import CheckpointIO
import numpy as np
import torch
import torch.distributed as dist
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
)
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.interface import AMPModelMixin, ModelWrapper, OptimizerWrapper
from colossalai.logging import get_dist_logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.multimodal_parallel_plugin.modal_parallel_plugin import (
    ModalParallelPlugin,
)
from cornstarch.plugin.pipeweaver_parallel_plugin.modal_process_group_mesh import (
    PipeweaverProcessGroupMesh,
)
from cornstarch.plugin.pipeweaver_parallel_plugin.pipeline_schedule import (
    PipeweaverEncoderTrainingPipeweaverScheduler,
)
from cornstarch.plugin.pipeweaver_parallel_plugin.pipeweaver_stage_manager import (
    PipeweaverPipelineStageManager,
)
from cornstarch.shardformer.shard.shard_config import ShardConfig

logger = logging.get_logger(__name__)


class PipeweaverParallelModule(ModelWrapper, AMPModelMixin):
    """Model wrapper for PipeWeaver co-located multimodal training.

    Each rank hosts both an encoder stage and an LLM stage.  The
    :attr:`stage_manager` ``current_mode`` attribute (set by the schedule)
    determines which sub-module is executed in a given :meth:`forward` call.

    Args:
        module: The :class:`~cornstarch.models.multimodal_language_model.MultimodalModel`
            to wrap.
        precision: Mixed-precision mode (``"fp16"``, ``"bf16"``, or ``"fp32"``).
        dp_group: Data-parallel process group.
        tp_group: Tensor-parallel process group.
        sp_group: Sequence-parallel process group.
        encoder_name: Name of the encoder modal (key in ``module.encoders``).
        encoder_shard_config: ShardConfig used when sharding the encoder.
        llm_shard_config: ShardConfig used when sharding the LLM.
    """

    def __init__(
        self,
        module: MultimodalModel,
        precision: str,
        dp_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        encoder_name: str,
        encoder_shard_config: ShardConfig,
        llm_shard_config: ShardConfig,
    ) -> None:
        assert isinstance(
            module, MultimodalModel
        ), f"Expected MultimodalModel, got {type(module)}"
        assert llm_shard_config is not None

        if (
            module.language_model.config.tie_word_embeddings
            and llm_shard_config.pipeline_template.num_stages > 1
        ):
            raise NotImplementedError(
                "Tied embeddings with pipeline parallelism cannot be synchronized."
            )

        self.stage_manager: PipeweaverPipelineStageManager = (
            llm_shard_config.pipeline_stage_manager
        )
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.use_ddp = False
        self.require_grad_sync = True
        self.shared_params = []
        self.shared_param_process_groups = []

        # PipeWeaver: each rank handles both modals.
        self._encoder_modal_name = f"{encoder_name}_encoder"
        self._llm_modal_name = "language_model"
        self.encoder_shard_config = encoder_shard_config
        self.llm_shard_config = llm_shard_config

        self.mixed_precision = None
        if precision == "fp16":
            self.mixed_precision = torch.float16
        elif precision == "bf16":
            self.mixed_precision = torch.bfloat16
        if self.mixed_precision is not None:
            module = module.to(self.mixed_precision)

        module = module.to(get_accelerator().get_current_device())
        super().__init__(module)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        module: MultimodalModel = self.module
        stage_manager = self.stage_manager
        current_mode = stage_manager.current_mode

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else module.language_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else module.language_model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else module.language_model.config.return_dict
        )

        if use_cache:
            logger.warning_once(
                "use_cache=True is not supported for pipeline models at the moment."
            )
            use_cache = False

        if current_mode == "llm":
            # ----------------------------------------------------------------
            # LLM forward
            # ----------------------------------------------------------------
            token_mask = torch.isin(
                input_ids,
                torch.tensor(list(module.token_ids.values()), device=input_ids.device),
            )
            labels_masked = labels.clone()
            labels_masked[token_mask] = -100

            if stage_manager.is_first_stage():
                # LLM first stage: merge encoder outputs into text embeddings.
                encoders_outputs: dict[str, tuple[torch.Tensor]] = {}
                modal_key = list(module.encoders.keys())[0]
                encoders_outputs[modal_key] = (hidden_states,)

                input_ids_masked = input_ids.clone()
                input_ids_masked[token_mask] = 0
                inputs_embeds = module.language_model.get_input_embeddings()(
                    input_ids_masked
                )
                inputs_embeds, attention_mask = module.merge_encoder_outputs(
                    encoders_outputs=encoders_outputs,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                language_model_inputs = dict(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    hidden_states=None,
                    labels=labels_masked,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                language_model_inputs.update(kwargs)

                if module.preprocess_llm_callback is not None:
                    callback_arguments = list(
                        inspect.signature(
                            module.preprocess_llm_callback
                        ).parameters.keys()
                    )
                    callback_inputs = {
                        key: value
                        for key, value in language_model_inputs.items()
                        if key in callback_arguments
                    }
                    callback_outputs = module.preprocess_llm_callback(**callback_inputs)
                    language_model_inputs.update(callback_outputs)
            else:
                assert inputs_embeds is None
                language_model_inputs = dict(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    hidden_states=hidden_states,
                    labels=labels_masked,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                language_model_inputs.update(kwargs)

            language_model_arguments = list(
                inspect.signature(module.language_model.forward).parameters.keys()
            )
            for key in list(language_model_inputs.keys()):
                if key not in language_model_arguments:
                    language_model_inputs.pop(key)

            result = module.language_model(**language_model_inputs)
            if isinstance(result, dict):
                result["attention_mask"] = attention_mask
            return result

        else:
            # ----------------------------------------------------------------
            # Encoder forward
            # ----------------------------------------------------------------
            encoder_name_base = self._encoder_modal_name.replace("_encoder", "")
            encoder_module = getattr(module, self._encoder_modal_name)

            encoder_inputs = {}
            if stage_manager.is_first_stage():
                if hidden_states is not None:
                    encoder_inputs["hidden_states"] = hidden_states
                encoder_inputs.update(
                    {
                        arg: kwargs[arg]
                        for arg in module.encoders_args[encoder_name_base]
                        if arg in kwargs
                    }
                )
            else:
                assert hidden_states is not None
                encoder_inputs["hidden_states"] = hidden_states

            for additional_arg in encoder_module.additional_args:
                if additional_arg in kwargs:
                    encoder_inputs[additional_arg] = kwargs[additional_arg]

            return encoder_module(
                **encoder_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def sync_shared_params(self) -> None:
        for shared_param, group in zip(
            self.shared_params, self.shared_param_process_groups
        ):
            if self.stage_manager.stage in shared_param:
                param = shared_param[self.stage_manager.stage]
                dist.all_reduce(param.grad, group=group)
            dist.barrier()

    @contextmanager
    def no_sync(self):
        old = self.require_grad_sync
        self.require_grad_sync = False
        try:
            if self.use_ddp:
                with self.module.no_sync():
                    yield
            else:
                yield
        finally:
            self.require_grad_sync = old

    def sync_dp_grads(self) -> None:
        if self.dp_group.size() == 1:
            return
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, group=self.dp_group)
                p.grad.div_(self.dp_group.size())

    def sync_sp_grads(self) -> None:
        pass

    def _hook_context(self):
        return nullcontext()

    def train(
        self,
        encoders_mode: dict[str, tuple[bool, bool]] = None,
        llm_mode: bool = True,
    ) -> PipeweaverParallelModule:
        self.module.train(encoders_mode, llm_mode)
        return self

    def set_modality_token_ids(
        self, token_ids: dict[str, int], new_num_tokens: int = 0
    ) -> None:
        self.module.set_modality_token_ids(token_ids, new_num_tokens)


class PipeweaverParallelPlugin(HybridParallelPlugin):
    """PipeWeaver plugin for co-located multimodal training.

    Each rank co-hosts one encoder PP stage and one LLM PP stage at the same
    pipeline index.  The :class:`PipeweaverEncoderTrainingPipeweaverScheduler`
    orchestrates a 3-phase execution:
      1. all encoder micro-batch forwards
      2. LLM 1F1B
      3. all encoder micro-batch backwards

    Both encoder and LLM must have the same PP / TP / SP / DP sizes.

    Args:
        encoder_plugin: Plugin (and ``PipelineTemplate``) for the encoder.
        encoder_name: Key under which the encoder appears in ``MultimodalModel``.
        language_model_plugin: Plugin for the language model.
        precision: ``"fp16"`` / ``"bf16"`` / ``"fp32"``.
        num_microbatches: Number of micro-batches per global batch.
        microbatch_size: Micro-batch size.
    """

    def __init__(
        self,
        encoder_plugin: ModalParallelPlugin,
        encoder_name: str,
        language_model_plugin: ModalParallelPlugin,
        precision: str = None,
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
    ) -> None:
        PipelinePluginBase.__init__(self)
        self.logger = get_dist_logger()

        assert (
            encoder_plugin.tp_size == language_model_plugin.tp_size
        ), "PipeWeaver requires encoder and LLM to share the same TP size."
        assert (
            encoder_plugin.sp_size == language_model_plugin.sp_size
        ), "PipeWeaver requires encoder and LLM to share the same SP size."
        assert (
            encoder_plugin.pipeline_template.num_stages
            == language_model_plugin.pipeline_template.num_stages
        ), "PipeWeaver requires encoder and LLM to have the same PP size."

        self.encoder_plugin = encoder_plugin
        self.encoder_name = encoder_name
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

        self.distributed_initialized: bool = False

    def __del__(self) -> None:
        pass

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

    def init_distributed(self) -> None:
        if self.distributed_initialized:
            return

        self.pg_mesh = PipeweaverProcessGroupMesh(
            encoder_template=self.encoder_plugin.pipeline_template,
            llm_template=self.language_model_plugin.pipeline_template,
            tp_size=self.encoder_plugin.tp_size,
            sp_size=self.encoder_plugin.sp_size,
        )
        self.stage_manager = PipeweaverPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )

        # Process groups — shared mesh, so TP/SP/DP groups are the same for
        # encoder and LLM.
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.sp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.sp_axis)
        self.pp_groups = self.pg_mesh.get_pp_groups()

        # Global PP group spans all pipeline stages for one DP replica.
        # In PipeWeaver, this equals the ordinary PP group.
        self.global_pp_group = self.pp_groups[0] if self.pp_groups else None
        self.pp_group = self.global_pp_group

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = self.pg_mesh.size(self.pg_mesh.pp_axis)

        self.scheduler = PipeweaverEncoderTrainingPipeweaverScheduler(
            self.stage_manager,
            self.num_microbatches,
            self.microbatch_size,
        )

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager
        self.shard_config.enable_tensor_parallelism = (
            dist.get_world_size(self.tp_group) > 1
        )
        self.shard_config.sequence_parallel_process_group = self.sp_group
        self.shard_config.enable_sequence_parallelism = (
            dist.get_world_size(self.sp_group) > 1
        )
        self.shard_config.sequence_parallelism_mode = (
            self.encoder_plugin.sequence_parallelism_mode
        )
        self.shard_config.__post_init__()

        # Sync gradients across DP * SP ranks when SP is enabled.
        if self.shard_config.enable_sequence_parallelism:
            self.dp_group = self.pg_mesh.get_group_along_axis(
                [self.pg_mesh.dp_axis, self.pg_mesh.sp_axis]
            )

        self.distributed_initialized = True

    def configure(
        self,
        model: MultimodalModel,
        optimizer: Optimizer | None = None,
        criterion: Callable[..., Any] | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> Tuple[
        nn.Module, OptimizerWrapper, Callable[..., Any], DataLoader, LRScheduler
    ]:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        param_info = get_param_info(optimizer)

        if not isinstance(model, ModelWrapper):
            # Shard encoder (set encoder mode so stage_manager.distribute_layers
            # returns encoder layers).
            self.stage_manager.set_encoder_mode()
            encoder_shard_config = replace(
                self.shard_config,
                pipeline_template=self.encoder_plugin.pipeline_template,
            )
            enc_module = model.get_submodule(f"{self.encoder_name}_encoder")
            enc_module = self.encoder_plugin.configure(
                enc_module, encoder_shard_config, self.stage_manager
            )
            model.add_module(f"{self.encoder_name}_encoder", enc_module)

            # Shard LLM (set llm mode).
            self.stage_manager.set_llm_mode()
            llm_shard_config = replace(
                self.shard_config,
                pipeline_template=self.language_model_plugin.pipeline_template,
                enable_flash_attention=False,
            )
            llm_module = model.get_submodule("language_model")
            llm_module = self.language_model_plugin.configure(
                llm_module, llm_shard_config, self.stage_manager
            )
            llm_module.config._attn_implementation = "bitfield_attention"
            model.add_module("language_model", llm_module)

            # Wrap in PipeweaverParallelModule.
            model = PipeweaverParallelModule(
                model,
                precision=self.precision,
                dp_group=self.dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                encoder_name=self.encoder_name,
                encoder_shard_config=encoder_shard_config,
                llm_shard_config=llm_shard_config,
            )

        if optimizer is not None:
            if not isinstance(optimizer, OptimizerWrapper):
                if self.precision in ["fp16", "bf16"]:
                    optimizer = HybridParallelAMPOptimizer(
                        optimizer,
                        model,
                        use_pipeline=self.enable_pipeline_parallelism,
                        param_info=param_info,
                        precision=self.precision,
                        max_norm=self.max_norm,
                        pp_process_group=self.global_pp_group,
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
                        pp_process_group=self.global_pp_group,
                        tp_process_group=self.tp_group,
                    )
                model.update_master_params = MethodType(
                    optimizer.update_master_params, model
                )

        return model, optimizer, criterion, dataloader, lr_scheduler

    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        sampler=None,
        **kwargs,
    ) -> DataLoader:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        _kwargs = kwargs.copy()

        def seed_worker(worker_id: int) -> None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

        if sampler is not None:
            # Custom batch sampler: inject dp_rank / dp_size and use as
            # batch_sampler so the caller controls partitioning.
            dp_size = self.pg_mesh.size(self.pg_mesh.dp_axis)
            dp_rank = self.pg_mesh.coordinate(self.pg_mesh.dp_axis)
            if hasattr(sampler, "num_replicas"):
                sampler.num_replicas = dp_size
            if hasattr(sampler, "rank"):
                sampler.rank = dp_rank
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                worker_init_fn=seed_worker,
                pin_memory=pin_memory,
                num_workers=num_workers,
                **_kwargs,
            )

        default_sampler = DistributedSampler(
            dataset,
            num_replicas=self.pg_mesh.size(self.pg_mesh.dp_axis),
            rank=self.pg_mesh.coordinate(self.pg_mesh.dp_axis),
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=default_sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> CheckpointIO:
        return None
