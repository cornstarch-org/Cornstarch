import inspect
from dataclasses import replace
from types import MethodType
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.accelerator import get_accelerator
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelAMPOptimizer,
    HybridParallelNaiveOptimizer,
    HybridParallelPlugin,
    get_param_info,
)
from colossalai.booster.plugin.pp_plugin_base import PipelinePluginBase
from colossalai.checkpoint_io import CheckpointIO
from colossalai.interface import AMPModelMixin, ModelWrapper, OptimizerWrapper
from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule
from colossalai.pipeline.stage_manager import PipelineStageManager
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.encoders_replicated_plugin.encoders_replicated_stage_manager import (
    EncodersReplicatedPipelineStageManager,
)
from cornstarch.plugin.encoders_replicated_plugin.process_group_mesh import (
    EncodersReplicatedProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
)

# from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
#     MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
# )
from cornstarch.shardformer.shard.shard_config import ShardConfig

logger = logging.get_logger(__name__)


class EncodersReplicatedMultimodalParallelModule(MultimodalParallelModule):
    def __init__(
        self,
        module: MultimodalModel,
        precision: str,
        dp_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        llm_shard_config: ShardConfig,
    ):
        assert isinstance(
            module, MultimodalModel
        ), f"Expected MultimodalModel, got {type(module)}"

        assert llm_shard_config is not None
        if (
            module.language_model.config.tie_word_embeddings
            and llm_shard_config.pipeline_template.num_stages > 1
        ):
            raise NotImplementedError(
                "Tied embeddings in pipeline parallelism cannot be synchronized as of now."
            )

        self.stage_manager = llm_shard_config.pipeline_stage_manager
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.use_ddp = False
        self.require_grad_sync = True
        self.shared_params = []  # TODO: add shared params
        self.shared_param_process_groups = []
        self.llm_shard_config = llm_shard_config

        # setting mixed_precision
        self.mixed_precision = None
        if precision == "fp16":
            self.mixed_precision = torch.float16
        elif precision == "bf16":
            self.mixed_precision = torch.bfloat16
        if self.mixed_precision is not None:
            module = module.to(self.mixed_precision)

        module = module.to(get_accelerator().get_current_device())

        ModelWrapper.__init__(self, module)
        AMPModelMixin.__init__(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        """
        Pipeline parallelism aware forward of `MultimodalModel.forward()`.
        """
        module: MultimodalModel = self.module
        stage_manager: PipelineStageManager = self.stage_manager

        if module.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError

        return_dict = (
            return_dict
            if return_dict is not None
            else module.language_model.config.return_dict
        )

        if output_attentions:
            logger.warning_once(
                "output_attentions=True is not supported for pipeline models at the moment."
            )
            output_attentions = False
        if output_hidden_states:
            logger.warning_once(
                "output_hidden_states=True is not supported for pipeline models at the moment."
            )
            output_hidden_states = False
        if use_cache:
            logger.warning_once(
                "use_cache=True is not supported for pipeline models at the moment."
            )
            use_cache = False

        # Run all encoders
        encoders_outputs = {}
        for modal_key, encoder_module in module.encoders.items():
            encoder_inputs = {
                arg: kwargs[arg]
                for arg in module.encoders_args[modal_key]
                if arg in kwargs
            }

            for additional_arg in encoder_module.additional_args:
                if additional_arg in kwargs:
                    encoder_inputs[additional_arg] = kwargs[additional_arg]

            encoders_outputs[modal_key] = encoder_module(
                **encoder_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run the language model
        if stage_manager.is_first_stage():
            inputs_embeds = module.language_model.get_input_embeddings()(input_ids)

            """
            Although encoders are replicated, here we consider tokens are injected
            in the middle of the inputs_embeds, not using cross-attention.
            With this scenario, we can run non cross-attention multimodal LLMs
            but overall computation may be inaccurate.
            It "simulates" the overhead of multiple encoder execution, not for
            the actual cross-attention.
            """

            # step 3. merge encoder outputs to llm inputs_embeds
            encoders_inputs: dict[str, dict] = {}
            # merging functions accept either BaseModelOutput or tuple of tensors,
            # and the first tensor (last_hidden_state) is merged.
            encoders_outputs_dict: dict[str, tuple[torch.Tensor]] = {}
            for modal_key, encoder_outputs in encoders_outputs.items():
                encoder_module: ModalEncoderModule = getattr(
                    module, f"{modal_key}_encoder"
                )
                encoder_inputs = {
                    arg: kwargs[arg]
                    for arg in module.encoders_args[modal_key]
                    if arg in kwargs
                }

                for additional_arg in encoder_module.additional_args:
                    if additional_arg in kwargs:
                        encoder_inputs[additional_arg] = kwargs[additional_arg]

                encoders_inputs[modal_key] = encoder_inputs
                encoders_outputs_dict[modal_key] = encoder_outputs

            inputs_embeds, attention_mask, position_ids, labels = (
                module.merge_encoder_outputs(
                    encoder_inputs=encoders_inputs,
                    encoder_outputs=encoders_outputs_dict,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

            language_model_inputs = dict(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                hidden_states=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            assert inputs_embeds is None

            language_model_inputs = dict(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                hidden_states=hidden_states,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # remove inputs that the language model doesn't accept
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


class EncodersReplicatedMultimodalParallelPlugin(HybridParallelPlugin):
    def __init__(
        self,
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

        self.language_model_plugin = language_model_plugin

        self.precision = precision
        self.zero_config = 0

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

    def __del__(self):
        pass

    def init_distributed(self):
        if self.distributed_initialized:
            return

        # self.pg_mesh = EncodersReplicatedProcessGroupMesh(self.language_model_plugin)
        pp_size, tp_size, sp_size = (
            self.language_model_plugin.pipeline_template.num_stages,
            self.language_model_plugin.tp_size,
            self.language_model_plugin.sp_size,
        )
        num_ranks_in_model = pp_size * tp_size * sp_size
        assert (
            dist.get_world_size() % num_ranks_in_model == 0
        ), f"The world size {dist.get_world_size()} must be divisible by num_ranks per replica {num_ranks_in_model}."
        self.pg_mesh = EncodersReplicatedProcessGroupMesh(
            llm_template=(
                self.language_model_plugin.pipeline_template,
                tp_size,
                sp_size,
            )
        )
        self.stage_manager = EncodersReplicatedPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.sp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.sp_axis)
        self.pp_groups = self.pg_mesh.get_group_along_axis(self.pg_mesh.pp_axis)
        self.p2p_group = self.pp_groups[0]

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = dist.get_world_size(group=self.pp_groups[0])

        self.schedule = OneForwardOneBackwardSchedule(
            self.stage_manager,
            self.num_microbatches,
            self.microbatch_size,
            enable_metadata_cache=False,
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
            self.language_model_plugin.sequence_parallelism_mode
            if self.shard_config.enable_sequence_parallelism
            else None
        )
        self.shard_config.__post_init__()
        self.distributed_initialized = True

    def configure(
        self,
        model: MultimodalModel,
        optimizer: Optimizer | None = None,
        criterion: Callable[..., Any] | None = None,
        dataloader: DataLoader | None = None,
        lr_scheduler: LRScheduler | None = None,
    ) -> tuple[
        nn.Module, OptimizerWrapper, Callable[..., Any], DataLoader, LRScheduler
    ]:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        param_info = get_param_info(optimizer)

        if not isinstance(model, ModelWrapper):
            # Configure to replace layers for compatibility
            # but without any parallelization
            tp_size = dist.get_world_size(self.tp_group)
            for encoder_module in model.encoders.values():
                encoder_plugin = ModalParallelPlugin(tp_size=tp_size)
                encoder_shard_config = ShardConfig(
                    tensor_parallel_process_group=self.tp_group,
                    enable_tensor_parallelism=tp_size > 1,
                )
                encoder_plugin.configure(
                    encoder_module, encoder_shard_config, self.stage_manager
                )

            llm_shard_config = replace(
                self.shard_config,
                pipeline_template=self.language_model_plugin.pipeline_template,
            )
            module = model.get_submodule("language_model")
            module = self.language_model_plugin.configure(
                module, llm_shard_config, self.stage_manager
            )
            model.add_module("language_model", module)

            model = EncodersReplicatedMultimodalParallelModule(
                model,
                precision=self.precision,
                dp_group=self.dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
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
                        pp_process_group=self.pp_groups[0],
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
                        pp_process_group=self.pp_groups[0],
                        tp_process_group=self.tp_group,
                    )
                # inject update_master_params
                model.update_master_params = MethodType(
                    optimizer.update_master_params, model
                )

        return model, optimizer, criterion, dataloader, lr_scheduler

    def get_checkpoint_io(self) -> CheckpointIO:
        from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_checkpoint_io import (
            MultimodalParallelCheckpointIO,
        )

        return MultimodalParallelCheckpointIO(self)
