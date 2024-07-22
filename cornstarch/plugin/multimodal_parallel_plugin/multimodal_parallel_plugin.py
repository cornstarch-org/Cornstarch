import inspect
from contextlib import contextmanager
from dataclasses import replace
from types import MethodType
from typing import Any, Callable, Optional, Tuple

from colossalai.checkpoint_io import CheckpointIO
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
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from cornstarch.models.multimodal_language_model import MultimodalModel
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
        self.stage_manager = llm_shard_config.pipeline_stage_manager
        self.dp_group = dp_group
        self.tp_group = tp_group
        self.sp_group = sp_group
        self.use_dpp = False
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
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        hidden_states: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Pipeline parallelism aware forward of `MultimodalModel.forward()`.
        """
        module: MultimodalModel = self.module
        stage_manager: MultiModalPipelineStageManager = self.stage_manager

        if module.language_model is None:
            # Does not support CLIP-like encoder only multimodal model yet
            raise NotImplementedError

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

        module: MultimodalModel = self.module
        if self.my_modal_name == "language_model":
            if stage_manager.is_first_stage(check_only_in_modal=True):
                # Forward in the first stage of the language model
                encoders_outputs = hidden_states
                encoders_attention_mask = torch.ones(
                    encoders_outputs.size()[:-1],
                    dtype=torch.long,
                    device=encoders_outputs.device,
                )

                # step 2. merge encoded multimodal features into text embeddings
                inputs_embeds = module.language_model.get_input_embeddings()(input_ids)
                inputs_embeds = torch.cat([encoders_outputs, inputs_embeds], dim=1)
                hidden_states = None

                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids).to(
                        encoders_attention_mask.device
                    )
                attention_mask = torch.cat(
                    [encoders_attention_mask, attention_mask], dim=1
                )
            else:
                assert inputs_embeds is None

            if hidden_states is not None:
                language_model_inputs = dict(
                    input_ids=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    hidden_states=hidden_states,
                )
            else:
                language_model_inputs = dict(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
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

            outputs = module.language_model(**language_model_inputs)

            if stage_manager.is_last_stage(check_only_in_modal=True):
                # TODO: add padding to labels and use LM's loss calculation
                #       Search "pad the labels" from Shiqi in the chat for context
                # TODO: support tensor parallelism as well
                logits = outputs.logits if return_dict else outputs[0]
                loss = None
                # we compute the loss here since we need to take into account the sequence length of the query embeds
                if labels is not None:
                    labels = labels.to(logits.device)
                    logits = logits[:, -labels.size(1) :, :]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous().to(logits.device)

                    # Flatten the tokens
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, module.language_model.config.vocab_size),
                        shift_labels.view(-1),
                    )

                if not return_dict:
                    output = (
                        logits,
                        past_key_values,
                        outputs.hidden_states,
                        outputs.attentions,
                    )
                    return ((loss,) + output) if loss is not None else output

                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            # LM already returns a dict {"hidden_states": tensor}
            return outputs
        elif "encoder" in self.my_modal_name:
            # TODO: support colocated modal forward.
            # assume currently they are parallelized.
            modal_key = self.my_modal_name.replace("_encoder", "")
            args = {
                arg: kwargs[arg]
                for arg in module.encoders_args[modal_key]
                if arg in kwargs
            }
            if "output_attentions" in module.encoders_args[modal_key]:
                args["output_attentions"] = output_attentions
            if "output_hidden_states" in module.encoders_args[modal_key]:
                args["output_hidden_states"] = output_hidden_states
            if "return_dict" in module.encoders_args[modal_key]:
                args["return_dict"] = return_dict

            encoder_module = getattr(module, self.my_modal_name)
            if hidden_states is not None:
                assert not stage_manager.is_first_stage(check_only_in_modal=True)
                args.pop(encoder_module.module.main_input_name)
                args["hidden_states"] = hidden_states

            outputs = encoder_module(**args)
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
            if self.use_dpp:
                # If using data parallel processing (use_dpp), disable synchronization too.
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
            ),
        )
        self.stage_manager = MultiModalPipelineStageManager(
            self.pg_mesh, self.pg_mesh.pp_axis
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
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
        self.shard_config.__post_init__()

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
                sp_group=None,
                encoder_shard_configs=encoder_shard_configs,
                llm_shard_config=llm_shard_config,
            )

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
        return None
