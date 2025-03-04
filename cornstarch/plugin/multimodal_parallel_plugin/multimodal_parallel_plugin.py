from __future__ import annotations

import inspect
import random
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import MethodType
from typing import Any, Callable, Optional, Tuple

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
from colossalai.checkpoint_io import CheckpointIO
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
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_parallel_plugin import (
    ModalParallelPlugin,
)
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_stage_manager import (
    MultiModalPipelineStageManager,
)
from cornstarch.shardformer.shard.shard_config import ShardConfig

logger = logging.get_logger(__name__)


class MultimodalParallelModule(ModelWrapper, AMPModelMixin):
    def __init__(
        self,
        module: MultimodalModel,
        precision: str,
        dp_group: dist.ProcessGroup,
        tp_group: dist.ProcessGroup,
        sp_group: dist.ProcessGroup,
        encoder_shard_configs: Optional[dict[str, ShardConfig]] = None,
        llm_shard_config: Optional[ShardConfig] = None,
        decoder_shard_configs: Optional[dict[str, ShardConfig]] = None,
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
        self.decoder_shard_configs = decoder_shard_configs

        # Cache my modal so that do forward only on the modal
        stage_manager: MultiModalPipelineStageManager = self.stage_manager
        my_modal_template = stage_manager.stage_index_to_modal[
            stage_manager.pg_mesh.coords[0][stage_manager.pipeline_axis]
        ]
        my_modal_name: str = None
        if my_modal_template in stage_manager.pg_mesh.encoder_templates.keys():
            my_modal_name = next(
                modal_name
                for modal_name, shard_config in encoder_shard_configs.items()
                if shard_config.pipeline_template == my_modal_template
            )
            my_modal_name = f"{my_modal_name}_encoder"
        elif my_modal_template == stage_manager.pg_mesh.llm_template[0]:
            my_modal_name = "language_model"
        elif my_modal_template in stage_manager.pg_mesh.decoder_templates.keys():
            my_modal_name = next(
                modal_name
                for modal_name, shard_config in decoder_shard_configs.items()
                if shard_config.pipeline_template == my_modal_template
            )
            my_modal_name = f"{my_modal_name}_decoder"
        assert (
            my_modal_name is not None
        ), f"Cannot find a modal module that rank {dist.get_rank()} owns."
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

        super().__init__(module)

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

        if "decoder" in self.my_modal_name:
            raise NotImplementedError("Decoder forward is not implemented yet.")

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

        if self.my_modal_name == "language_model":
            token_mask = torch.isin(
                input_ids,
                torch.tensor(list(module.token_ids.values()), device=input_ids.device),
            )
            labels_masked = labels.clone()
            labels_masked[token_mask] = -100

            if stage_manager.is_first_stage(check_only_in_modal=True):
                # Forward in the first stage of the language model

                # merging functions accept either BaseModelOutput or tuple of tensors,
                # and the first tensor (last_hidden_state) is merged.
                encoders_outputs: dict[str, tuple[torch.Tensor]] = {}

                for encoder_outputs, modal_key in zip(
                    hidden_states, module.encoders.keys()
                ):
                    encoder_module: ModalEncoderModule = getattr(
                        module, f"{modal_key}_encoder"
                    )
                    encoders_outputs[modal_key] = (encoder_outputs,)

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
        elif "encoder" in self.my_modal_name:
            # It assumes currently they are parallelized.
            # For colocated model forward, use MultimodalSequentialPlugin.
            modal_key = self.my_modal_name.replace("_encoder", "")
            encoder_module = getattr(module, self.my_modal_name)

            if stage_manager.is_first_stage(check_only_in_modal=True):
                assert hidden_states is None
                encoder_inputs = {
                    arg: kwargs[arg]
                    for arg in module.encoders_args[modal_key]
                    if arg in kwargs
                }

                for additional_arg in encoder_module.additional_args:
                    if additional_arg in kwargs:
                        encoder_inputs[additional_arg] = kwargs[additional_arg]
            else:
                assert hidden_states is not None
                encoder_inputs = dict(hidden_states=hidden_states)

            outputs = encoder_module(
                **encoder_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            return outputs
        elif "decoder" in self.my_modal_name:
            raise NotImplementedError()

    def sync_shared_params(self):
        for shared_param, group in zip(
            self.shared_params, self.shared_param_process_groups
        ):
            if self.stage_manager.stage in shared_param:
                param = shared_param[self.stage_manager.stage]
                dist.all_reduce(param.grad, group=group)
            dist.barrier()

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable automatic gradient synchronization (all-reduce) and allow manual synchronization
        when 'no_sync' is active. Alternatively, synchronization will occur in the first forward-backward pass
        when exiting the context.
        """

        # Store the current value of 'require_grad_sync' to restore it later.
        old_require_grad_sync = self.require_grad_sync
        # Disable automatic gradient synchronization.
        self.require_grad_sync = False
        try:
            if self.use_ddp:
                # If using data parallel processing (use_ddp), disable synchronization too.
                with self.module.no_sync():
                    yield
            else:
                yield
        finally:
            # Restore the original value of 'require_grad_sync'.
            self.require_grad_sync = old_require_grad_sync

    def sync_dp_grads(self):
        r"""
        Synchronize gradients across data parallelism (DP) if the DP group size is greater than 1.
        This function performs an all-reduce operation to combine gradients from different devices in the DP group.

        Args:
            None

        Returns:
            None
        """

        # Check if the DP group size is 1, meaning no synchronization is needed.
        if self.dp_group.size() == 1:
            return

        # Iterate through the model's parameters and perform gradient synchronization.
        for p in self.module.parameters():
            if p.grad is not None:
                # Perform all-reduce to combine gradients from different devices.
                dist.all_reduce(p.grad, group=self.dp_group)
                # Normalize the gradient by dividing it by the DP group size.
                p.grad.div_(self.dp_group.size())

    def sync_sp_grads(self):
        # For context parallelisms that Cornstarch multimodal module supports (all_to_all and ring_attn),
        # ranks have the entire copy of model parameters.
        # Therefore, sp ranks are grouped in the dp group and
        # synchronization for sp gradients is done within dp_group at once.
        # see the last part of MultimodalParallelPlugin.init_distributed()
        # that recreates dp_group with dp * sp ranks.
        # No need to synchronize gradients again here.
        pass

    def _hook_context(self):
        return nullcontext()

    def train(
        self, encoders_mode: dict[str, tuple[bool, bool]] = None, llm_mode=True
    ) -> MultimodalParallelModule:
        self.module.train(encoders_mode, llm_mode)
        return self

    def set_modality_token_ids(
        self, token_ids: dict[str, int], new_num_tokens: int = 0
    ):
        module: MultimodalModel = self.module
        module.set_modality_token_ids(token_ids, new_num_tokens)


class MultimodalParallelPlugin(HybridParallelPlugin):
    """Plugin for multimodal language model.
    Tensor parallel, pipeline parallel, and data parallel are combined in this plugin.
    Each modal has its own parallel configuration defined in ModalParallelPlugin.
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
        self.logger = get_dist_logger()
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

        self.distributed_initialized: bool = False

    def __del__(self):
        pass

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
        if self.distributed_initialized:
            return

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
        # Note that current schedule is encoder-llm only.
        # Decoder-llm needs another schedule, and encoder-decoder cannot be trained together.
        # TODO: implement interleaved parallelism to train encoder and decoder at the same time.

        self.pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={
                plugin.pipeline_template: plugin.tp_size
                for plugin in self.encoder_plugins.values()
            },
            llm_template=(
                self.language_model_plugin.pipeline_template,
                self.language_model_plugin.tp_size,
                self.language_model_plugin.sp_size,
            ),
        )
        self.stage_manager = MultiModalPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.sp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.sp_axis)
        self.pp_groups = self.pg_mesh.get_group_along_axis(self.pg_mesh.pp_axis)

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = dist.get_world_size(group=self.pp_groups[0])

        # TODO: implement a new one if needed!
        self.schedule = MultimodalEncoderTrainingOneForwardOneBackwardSchedule(
            self.stage_manager, self.num_microbatches, self.microbatch_size
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
    ) -> Tuple[
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
                )
                module = model.get_submodule(f"{modal_name}_encoder")
                module = encoder.configure(module, shard_config, self.stage_manager)
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
            model.add_module("language_model", module)

            model = MultimodalParallelModule(
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

    def prepare_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 1024,
        drop_last: bool = False,
        pin_memory: bool = False,
        num_workers: int = 0,
        distributed_sampler_cls=None,
        **kwargs,
    ):
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        _kwargs = kwargs.copy()
        distributed_sampler_cls = distributed_sampler_cls or DistributedSampler
        sampler = distributed_sampler_cls(
            dataset,
            num_replicas=self.pg_mesh.size(self.pg_mesh.dp_axis),
            rank=self.pg_mesh.coords[0][self.pg_mesh.dp_axis],
            shuffle=shuffle,
        )

        # Deterministic dataloader
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            worker_init_fn=seed_worker,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> CheckpointIO:
        from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_checkpoint_io import (
            MultimodalParallelCheckpointIO,
        )

        return MultimodalParallelCheckpointIO(self)
