import inspect
from collections import OrderedDict
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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)
from cornstarch.plugin.encoders_colocated_plugin.encoders_colocated_stage_manager import (
    EncodersColocatedPipelineStageManager,
)
from cornstarch.plugin.encoders_colocated_plugin.one_f_one_b import (
    MultimodalColocatedOneForwardOneBackwardSchedule,
)
from cornstarch.plugin.encoders_colocated_plugin.process_group_mesh import (
    EncodersColocatedProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)
from cornstarch.shardformer.shard.shard_config import ShardConfig

logger = logging.get_logger(__name__)


class EncodersColocatedMultimodalParallelModule(MultimodalParallelModule):
    def __init__(
        self,
        module: MultimodalModel,
        precision: str,
        dp_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        encoder_shard_configs: dict[str, ShardConfig],
        llm_shard_config: ShardConfig,
    ):
        assert isinstance(
            module, MultimodalModel
        ), f"Expected MultimodalModel, got {type(module)}"

        # stage manager is also in all shard configs, but they all have the same
        # stage manager, but only different pipeline templates.
        # TODO: if llm_shard_config is None, use another shard_config
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
        self.encoder_shard_configs = encoder_shard_configs
        self.llm_shard_config = llm_shard_config

        # Cache my modal so that do forward only on the modal
        stage_manager: MultiModalPipelineStageManager = self.stage_manager
        my_modal_template = stage_manager.stage_index_to_modal[
            stage_manager.pg_mesh.coords[0][stage_manager.pipeline_axis]
        ]
        my_modal_name: list[str] | str = None
        if my_modal_template == stage_manager.pg_mesh.llm_template[0]:
            my_modal_name = "language_model"
        else:
            my_modal_name = list(module.encoders.keys())
        self.my_modal_name = my_modal_name

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
        """
        Pipeline parallelism aware forward of `MultimodalModel.forward()`.
        """
        module: MultimodalModel = self.module
        stage_manager: MultiModalPipelineStageManager = self.stage_manager

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

        assert "decoder" not in self.my_modal_name, "TODO: decoder forward"

        if self.my_modal_name == "language_model":
            token_mask = torch.isin(
                input_ids,
                torch.tensor(list(module.token_ids.values()), device=input_ids.device),
            )
            labels_masked = labels.clone()
            labels_masked[token_mask] = -100

            if stage_manager.is_first_stage(check_only_in_modal=True):

                encoders_outputs: dict[str, tuple[torch.Tensor]] = {}

                for modal_key in module.encoders.keys():
                    if f"encoder_outputs_{modal_key}" not in kwargs:
                        continue
                    # make a tuple so that in merge_encoder_outputs the actual values are taken
                    encoders_outputs[modal_key] = (
                        kwargs.pop(f"encoder_outputs_{modal_key}"),
                    )

                # step 2. merge encoded multimodal features into text embeddings
                # mask out special tokens from input_ids to avoid out of index error
                # and use it as an input to embedding.
                input_ids_masked = input_ids.clone()
                input_ids_masked[token_mask] = 0
                inputs_embeds = module.language_model.get_input_embeddings()(
                    input_ids_masked
                )

                # step 3. merge encoder outputs to llm inputs_embeds
                inputs_embeds, attention_mask = module.merge_encoder_outputs(
                    encoders_outputs=encoders_outputs,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )

                # step 4. run llm with merged inputs_embeds
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
                    # filter out inputs that the preprocess_llm_callback doesn't accept
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

            # remove inputs that the language model doesn't accept
            language_model_arguments = list(
                inspect.signature(module.language_model.forward).parameters.keys()
            )
            for key in list(language_model_inputs.keys()):
                if key not in language_model_arguments:
                    language_model_inputs.pop(key)

            result = module.language_model(**language_model_inputs)

            # bitfield attention mask cannot be generated in the following stages.
            # Add attention mask to the result.
            if isinstance(result, dict):
                result["attention_mask"] = attention_mask
            return result
        else:
            assert isinstance(self.my_modal_name, list)
            encoders_outputs = {}

            if stage_manager.is_first_stage(check_only_in_modal=True):
                for modal_key, encoder_module in module.encoders.items():
                    assert hidden_states is None
                    encoder_inputs = {
                        arg: kwargs[arg]
                        for arg in module.encoders_args[modal_key]
                        if arg in kwargs
                    }

                    for additional_arg in encoder_module.additional_args:
                        if additional_arg in kwargs:
                            encoder_inputs[additional_arg] = kwargs[additional_arg]

                    encoder_output = encoder_module(
                        **encoder_inputs,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )
                    encoders_outputs[modal_key] = encoder_output
            else:
                for modal_key, encoder_module in module.encoders.items():
                    if f"encoder_outputs_{modal_key}" not in kwargs:
                        continue

                    encoder_inputs = kwargs[f"encoder_outputs_{modal_key}"]

                    for additional_arg in encoder_module.additional_args:
                        if additional_arg in kwargs:
                            encoder_inputs[additional_arg] = kwargs[additional_arg]

                    encoders_outputs[modal_key] = encoder_module(
                        **encoder_inputs,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                    )

            # merge hidden_states tensors into a list.
            return {
                f"encoder_outputs_{modal_key}": (
                    encoder_outputs[0]
                    if isinstance(encoder_outputs, OrderedDict)
                    else encoder_outputs
                )
                for modal_key, encoder_outputs in encoders_outputs.items()
            }


class EncodersColocatedMultimodalParallelPlugin(HybridParallelPlugin):
    """Plugin for multimodal language model.
    Unlike `MultimodalParallelPlugin`, this plugin is designed to mimic
    existing chain-like Megatron-LM style pipeline parallelism, where
    pipeline stages are executed sequentailly without DAG-like scheduling.

    Encoders may have different `ModalParallelPlugin` instances, but their
    parallelization configuration (e.g. tp_size, sp_size, sequence_parallelism_mode,
    number of pipeline stages, etc) must be the same.
    """

    def __init__(
        self,
        encoder_plugins: dict[str, ModalParallelPlugin] = None,
        language_model_plugin: ModalParallelPlugin | None = None,
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
    ):
        PipelinePluginBase.__init__(self)

        first_plugin = next(iter(encoder_plugins.values()))
        for plugin in encoder_plugins.values():
            assert (
                first_plugin.tp_size == plugin.tp_size
            ), "All encoder plugins must have the same number of tensor parallel degree."
            assert (
                first_plugin.sp_size == plugin.sp_size
            ), "All encoder plugins must have the same number of sequence parallel degree."
            assert (
                first_plugin.sequence_parallelism_mode
                == plugin.sequence_parallelism_mode
            ), "All encoder plugins must have the same sequence parallelism mode."
            assert (
                first_plugin.pipeline_template.num_stages
                == plugin.pipeline_template.num_stages
            ), "All encoder plugins must have the same number of stages."

        self.encoder_plugins = encoder_plugins
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

        self.pg_mesh = EncodersColocatedProcessGroupMesh(
            encoder_templates={
                plugin.pipeline_template: (plugin.tp_size, plugin.sp_size)
                for plugin in self.encoder_plugins.values()
            },
            llm_template=(
                self.language_model_plugin.pipeline_template,
                self.language_model_plugin.tp_size,
                self.language_model_plugin.sp_size,
            ),
        )
        self.stage_manager = EncodersColocatedPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.sp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.sp_axis)
        self.pp_groups = self.pg_mesh.get_group_along_axis(self.pg_mesh.pp_axis)
        self.p2p_group = self.pp_groups[0]

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = dist.get_world_size(group=self.pp_groups[0])

        self.schedule = MultimodalColocatedOneForwardOneBackwardSchedule(
            self.stage_manager, self.num_microbatches, self.microbatch_size
        )

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager
        self.shard_config.enable_tensor_parallelism = (
            dist.get_world_size(self.tp_group) > 1
        )

        target_plugin: ModalParallelPlugin
        my_modal_template = self.stage_manager.stage_index_to_modal[
            self.stage_manager.pg_mesh.coords[0][self.stage_manager.pipeline_axis]
        ]
        if my_modal_template == self.stage_manager.pg_mesh.llm_template[0]:
            target_plugin = self.language_model_plugin
        else:
            target_plugin = list(self.encoder_plugins.values())[0]

        self.shard_config.sequence_parallel_process_group = self.sp_group
        self.shard_config.enable_sequence_parallelism = (
            dist.get_world_size(self.sp_group) > 1
        )
        self.shard_config.sequence_parallelism_mode = (
            target_plugin.sequence_parallelism_mode
        )
        self.shard_config.__post_init__()

        # sync gradients across DP * SP ranks
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
    ) -> tuple[
        nn.Module, OptimizerWrapper, Callable[..., Any], DataLoader, LRScheduler
    ]:
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        param_info = get_param_info(optimizer)

        if not isinstance(model, ModelWrapper):
            encoder_shard_configs = {}
            for modal_name, encoder in self.encoder_plugins.items():
                shard_config = replace(
                    self.shard_config,
                    pipeline_template=encoder.pipeline_template,
                    pipeline_stage_manager=self.stage_manager.encoder_stage_managers[
                        encoder.pipeline_template
                    ],
                )
                module = model.get_submodule(f"{modal_name}_encoder")
                module = encoder.configure(
                    module,
                    shard_config,
                    self.stage_manager.encoder_stage_managers[
                        encoder.pipeline_template
                    ],
                )
                model.add_module(f"{modal_name}_encoder", module)
                encoder_shard_configs[modal_name] = shard_config

            llm_shard_config = replace(
                self.shard_config,
                pipeline_template=self.language_model_plugin.pipeline_template,
                enable_flash_attention=False,
            )
            module = model.get_submodule("language_model")
            module = self.language_model_plugin.configure(
                module, llm_shard_config, self.stage_manager
            )
            module.config._attn_implementation = "bitfield_attention"
            model.add_module("language_model", module)

            model = EncodersColocatedMultimodalParallelModule(
                model,
                precision=self.precision,
                dp_group=self.dp_group,
                tp_group=self.tp_group,
                sp_group=self.sp_group,
                encoder_shard_configs=encoder_shard_configs,
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
