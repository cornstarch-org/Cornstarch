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

from cornstarch.plugin.multimodal_parallel_plugin.global_batch_reorder_sampler import (  # noqa: F401  # re-exported for callers
    GlobalBatchReorderSampler,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.multimodal_parallel_plugin.modal_parallel_plugin import (
    ModalParallelPlugin,
)
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b import (
    MultimodalEncoderTrainingOneForwardOneBackwardSchedule,
)
from cornstarch.plugin.multimodal_parallel_plugin.multimodal_zbpp import (
    MultimodalEncoderTrainingZeroBubblePipelineSchedule,
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
        self.my_modal_name = self._resolve_my_modal_name()

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

    def _resolve_my_modal_name(self) -> str:
        """Determine which modal sub-module this rank owns.

        Uses ``self.stage_manager``, ``self.encoder_shard_configs``, and
        ``self.decoder_shard_configs`` so the result is always consistent with
        the latest reconfiguration.  Called from ``__init__`` and from Phase 9
        of :meth:`MultimodalParallelPlugin.reconfigure` after shard configs
        have been refreshed.
        """
        stage_manager: MultiModalPipelineStageManager = self.stage_manager
        my_modal_template = stage_manager.pg_mesh.my_modal
        my_modal_name: Optional[str] = None

        if my_modal_template in stage_manager.pg_mesh.encoder_templates.keys():
            my_modal_name = next(
                modal_name
                for modal_name, shard_config in self.encoder_shard_configs.items()
                if shard_config.pipeline_template == my_modal_template
            )
            my_modal_name = f"{my_modal_name}_encoder"
        elif my_modal_template == stage_manager.pg_mesh.llm_template[0]:
            my_modal_name = "language_model"
        elif (
            self.decoder_shard_configs is not None
            and my_modal_template in stage_manager.pg_mesh.decoder_templates.keys()
        ):
            my_modal_name = next(
                modal_name
                for modal_name, shard_config in self.decoder_shard_configs.items()
                if shard_config.pipeline_template == my_modal_template
            )
            my_modal_name = f"{my_modal_name}_decoder"

        assert (
            my_modal_name is not None
        ), f"Cannot find a modal module that rank {dist.get_rank()} owns."
        return my_modal_name

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

                encoders_outputs: dict[str, tuple[torch.Tensor]] = {}
                modal_key = list(module.encoders.keys())[0]
                encoders_outputs[modal_key] = (hidden_states,)

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
        elif "encoder" in self.my_modal_name:
            # It assumes currently they are parallelized.
            # For colocated model forward, use MultimodalSequentialPlugin.
            modal_key = self.my_modal_name.replace("_encoder", "")
            encoder_module = getattr(module, self.my_modal_name)

            encoder_inputs = {}
            if stage_manager.is_first_stage(check_only_in_modal=True):
                if hidden_states is not None:
                    encoder_inputs["hidden_states"] = hidden_states

                encoder_inputs.update(
                    {
                        arg: kwargs[arg]
                        for arg in module.encoders_args[modal_key]
                        if arg in kwargs
                    }
                )

            else:
                assert hidden_states is not None
                encoder_inputs["hidden_states"] = hidden_states

            for additional_arg in encoder_module.additional_args:
                if additional_arg in kwargs:
                    encoder_inputs[additional_arg] = kwargs[additional_arg]

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


def _update_optimizer_param_groups(
    optimizer,
    rank: int,
    source_ownership,
    target_ownership,
    param_snapshot: dict,
    model: nn.Module,
) -> None:
    """Reconcile ``optimizer.param_groups`` with parameter ownership changes.

    After :meth:`ReconfigurationExecutor.execute` runs:

    * **Ranks that lose ownership** (owner → placeholder): remove the stale
      parameter from every inner param group so the optimizer no longer
      iterates over gradient-free orphans.
    * **Ranks that gain ownership** (placeholder → owner): add the freshly
      created ``nn.Parameter`` to the first inner param group.  This step is
      skipped for AMP optimizers (``HybridParallelAMPOptimizer``) because
      adding a working param to the master-param-keyed groups requires creating
      an fp32 master copy; the optimizer state entry is still written by
      :meth:`redistribute_optimizer_states`.

    Args:
        optimizer: ColossalAI ``OptimizerWrapper`` (or plain ``Optimizer``).
        rank: Current process rank.
        source_ownership: Ownership map before redistribution.
        target_ownership: Ownership map after redistribution.
        param_snapshot: ``{name: nn.Parameter}`` captured before execute().
        model: The ``MultimodalParallelModule`` whose parameters were updated.
    """
    from cornstarch.reconfiguration.utils import get_param_by_name as _gpbn

    def _owned_names(ownership, r):
        if r not in ownership:
            return set()
        own = ownership[r]
        return {n for n in own.layer_names if not own.is_placeholder.get(n, False)}

    source_owners = _owned_names(source_ownership, rank)
    target_owners = _owned_names(target_ownership, rank)
    newly_owned = target_owners - source_owners
    newly_lost = source_owners - target_owners

    if not newly_owned and not newly_lost:
        return

    # Detect AMP optimizer (master-param-keyed state).
    try:
        is_amp = optimizer.get_master_to_working_map() is not None
    except AttributeError:
        is_amp = False

    inner_groups = optimizer.optim.param_groups

    # Remove newly-lost parameters from every inner param group.
    if newly_lost:
        lost_ids: set = {
            id(param_snapshot[n]) for n in newly_lost if n in param_snapshot
        }
        for group in inner_groups:
            group["params"] = [p for p in group["params"] if id(p) not in lost_ids]

    # Add newly-owned parameters to the first inner param group (non-AMP only).
    if newly_owned and not is_amp and inner_groups:
        new_params = []
        for name in sorted(newly_owned):
            try:
                new_params.append(_gpbn(model, name))
            except KeyError:
                pass
        inner_groups[0]["params"].extend(new_params)


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
        pipeline_schedule: str = "1f1b",
    ):
        PipelinePluginBase.__init__(self)
        self.logger = get_dist_logger()
        assert encoder_plugins is not None and len(encoder_plugins) == 1, (
            "MultimodalParallelPlugin requires exactly one encoder plugin, "
            f"got {len(encoder_plugins) if encoder_plugins else 0}."
        )
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
        self.pipeline_schedule = pipeline_schedule.lower()
        if self.pipeline_schedule not in {"1f1b", "zbpp"}:
            raise ValueError(
                "pipeline_schedule must be one of {'1f1b', 'zbpp'}, "
                f"got {pipeline_schedule!r}."
            )

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
        self._init_pg_mesh()
        self._init_communication_groups()
        self.distributed_initialized = True

    def _init_pg_mesh(self):
        """Create the MultiModalProcessGroupMesh from current plugin configs."""
        self.pg_mesh = MultiModalProcessGroupMesh(
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

    def _init_communication_groups(self):
        """Create stage manager, process groups, scheduler and update shard config.

        Called both from :meth:`init_distributed` and from :meth:`reconfigure`
        (where ``self.pg_mesh`` has already been set to the new mesh).
        """
        self.stage_manager = MultiModalPipelineStageManager(
            self.pg_mesh,
            self.pg_mesh.pp_axis,
            use_zbv=(self.pipeline_schedule == "zbpp"),
        )
        self.dp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.dp_axis)
        self.tp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.tp_axis)
        self.sp_group = self.pg_mesh.get_group_along_axis(self.pg_mesh.sp_axis)
        self.pp_groups = self.pg_mesh.get_group_along_axis(self.pg_mesh.pp_axis)
        # Global PP group spans all pipeline stages across all modals in one DP replica.
        # Used by the optimizer for gradient-norm computation.
        self.global_pp_group = self.pg_mesh.get_global_pp_group()

        self.dp_size = dist.get_world_size(group=self.dp_group)
        self.pp_size = dist.get_world_size(group=self.pp_groups[0])

        enc_plugin = list(self.encoder_plugins.values())[0]
        encoder_sp_gather = (
            enc_plugin.sequence_parallelism_mode == "ring_attn"
            and enc_plugin.sp_size > 1
        )
        # Scatter the encoder's final TP layer output along the hidden
        # dimension only when the encoder has more TP ranks than the LLM
        # (TP fan-in case).  When enc_tp == llm_tp no deduplication happens
        # and the LLM expects a full (seq, H) tensor.
        tp_hidden_scatter = enc_plugin.tp_size > self.language_model_plugin.tp_size
        schedule_cls = (
            MultimodalEncoderTrainingOneForwardOneBackwardSchedule
            if self.pipeline_schedule == "1f1b"
            else MultimodalEncoderTrainingZeroBubblePipelineSchedule
        )
        self.scheduler = schedule_cls(
            self.stage_manager,
            self.num_microbatches,
            self.microbatch_size,
            encoder_sp_gather=encoder_sp_gather,
            encoder_tp_hidden_scatter=tp_hidden_scatter,
        )
        self._encoder_tp_scatter = tp_hidden_scatter

        self.shard_config.tensor_parallel_process_group = self.tp_group
        self.shard_config.pipeline_stage_manager = self.stage_manager
        self.shard_config.enable_tensor_parallelism = (
            dist.get_world_size(self.tp_group) > 1
        )

        my_modal = self.stage_manager.pg_mesh.my_modal
        if my_modal == self.stage_manager.pg_mesh.llm_template[0]:
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

    def reconfigure(
        self,
        new_encoder_plugins: dict,
        new_language_model_plugin,
        model,
        optimizer=None,
    ) -> None:
        """Reconfigure the plugin for a new parallel topology.

        Redistributes model parameters and optimizer states across the cluster
        to match the new configuration, then rebuilds all process groups.

        Preconditions:
        - ``dist.is_initialized()`` must be True.
        - No gradients exist (call after ``optimizer.step()``).
        - ``init_distributed()`` (and ``configure()``) have already been called.

        Args:
            new_encoder_plugins: New ``{name: ModalParallelPlugin}`` mapping.
            new_language_model_plugin: New LLM ``ModalParallelPlugin``.
            model: The wrapped ``MultimodalParallelModule``.
            optimizer: Optional ColossalAI ``OptimizerWrapper``.
        """
        from cornstarch.reconfiguration.executor import ReconfigurationExecutor
        from cornstarch.reconfiguration.ownership_analyzer import (
            TensorOwnershipAnalyzer,
            build_target_ownership,
        )

        assert dist.is_initialized(), "torch.distributed must be initialized."
        assert self.distributed_initialized, "Call init_distributed() first."

        # Phase 1: Analyze source ownership.  Pass the current TP group so
        # that TP-sharded parameters (Linear1D_Col / Linear1D_Row) are
        # annotated with their exact shard ranges rather than treated as full
        # tensors.  No gather step is needed.
        source_ownership = TensorOwnershipAnalyzer(model).analyze(
            tp_group=self.tp_group
        )

        # Phase 2: Build new pg_mesh (topology only — no dist.new_group calls
        # yet).  Immediately seed its cache from the old mesh so that unchanged
        # rank-set groups are reused rather than recreated.
        new_pg_mesh = MultiModalProcessGroupMesh(
            encoder_templates={
                plugin.pipeline_template: (plugin.tp_size, plugin.sp_size)
                for plugin in new_encoder_plugins.values()
            },
            llm_template=(
                new_language_model_plugin.pipeline_template,
                new_language_model_plugin.tp_size,
                new_language_model_plugin.sp_size,
            ),
        )
        new_pg_mesh.inherit_groups_from(self.pg_mesh)

        # Phase 3: Compute target ownership from new topology.  Pass
        # source_ownership so shard_dim and full-param sizes are propagated
        # to the target shard-range annotations (no extra communication).
        target_ownership = build_target_ownership(
            model,
            new_pg_mesh,
            new_encoder_plugins,
            new_language_model_plugin,
            source_ownership=source_ownership,
        )

        # Phase 4 (pre): Capture parameter snapshot before execute() replaces
        # parameter objects so that optimizer-state lookup by identity still works.
        param_snapshot = {n: p for n, p in model.named_parameters()}

        # Phase 4: Redistribute model parameters with direct shard-to-shard
        # all-to-all — no intermediate full-tensor assembly.
        executor = ReconfigurationExecutor(model)
        executor.execute(source_ownership, target_ownership)

        # Phase 4.5: Reconcile optimizer.param_groups with the ownership
        # changes that execute() just made.  Newly-owned parameters (whose
        # nn.Parameter objects were just created by set_param_by_name) must be
        # added to the optimizer so gradient updates flow through them.
        # Newly-lost parameters (now TensorPlaceholders) are removed to avoid
        # iterating gradient-free orphans in optimizer.step().
        if optimizer is not None:
            _update_optimizer_param_groups(
                optimizer,
                dist.get_rank(),
                source_ownership,
                target_ownership,
                param_snapshot,
                model,
            )

        # Phase 5: Redistribute optimizer states (same pattern).
        if optimizer is not None:
            executor.redistribute_optimizer_states(
                optimizer, source_ownership, target_ownership,
                param_snapshot=param_snapshot,
            )

        # Phase 6: Update plugin configuration.
        self.encoder_plugins = new_encoder_plugins
        self.language_model_plugin = new_language_model_plugin

        # Phase 7: Install the new pg_mesh and rebuild stage manager / groups /
        # shard config / scheduler.  _init_communication_groups() will call
        # dist.new_group() only for rank sets that are genuinely new (i.e. not
        # already in new_pg_mesh._ranks_to_group from inherit_groups_from).
        old_pg_mesh = self.pg_mesh
        self.pg_mesh = new_pg_mesh
        self._init_communication_groups()

        # Phase 8 (deferred): Now that new groups are established, destroy only
        # the groups the new configuration no longer needs.  Groups reused by
        # new_pg_mesh are kept alive.  No additional barrier is needed here:
        # the collective dist.new_group() calls in _init_communication_groups()
        # already provided the necessary synchronisation across all ranks.
        old_pg_mesh.destroy_stale_groups(set(new_pg_mesh._ranks_to_group))

        # Phase 9: Update live references on the model wrapper.
        model.dp_group = self.dp_group
        model.tp_group = self.tp_group
        model.sp_group = self.sp_group
        model.stage_manager = self.stage_manager

        # Fix B: Refresh the shard-config copies stored on the model wrapper.
        # These are independent dataclass instances created during configure();
        # update pipeline_template, stage_manager, and process-group fields so
        # they reflect the new topology.  _resolve_my_modal_name() (below)
        # reads pipeline_template, so this must happen before Fix A.
        if model.encoder_shard_configs is not None:
            updated_enc_cfgs = {}
            for modal_name, sc in model.encoder_shard_configs.items():
                new_enc_plugin = self.encoder_plugins[modal_name]
                updated_enc_cfgs[modal_name] = replace(
                    sc,
                    pipeline_template=new_enc_plugin.pipeline_template,
                    pipeline_stage_manager=self.stage_manager,
                    tensor_parallel_process_group=self.tp_group,
                    sequence_parallel_process_group=self.sp_group,
                )
            model.encoder_shard_configs = updated_enc_cfgs

        if model.llm_shard_config is not None:
            model.llm_shard_config = replace(
                model.llm_shard_config,
                pipeline_template=self.language_model_plugin.pipeline_template,
                pipeline_stage_manager=self.stage_manager,
                tensor_parallel_process_group=self.tp_group,
                sequence_parallel_process_group=self.sp_group,
            )

        if model.decoder_shard_configs is not None:
            updated_dec_cfgs = {}
            for modal_name, sc in model.decoder_shard_configs.items():
                updated_dec_cfgs[modal_name] = replace(
                    sc,
                    pipeline_stage_manager=self.stage_manager,
                    tensor_parallel_process_group=self.tp_group,
                    sequence_parallel_process_group=self.sp_group,
                )
            model.decoder_shard_configs = updated_dec_cfgs

        # Fix A: Recompute which modal sub-module this rank owns.  The cached
        # my_modal_name from construction may be stale if reconfiguration moved
        # ranks between encoder / LLM / decoder modals.
        model.my_modal_name = model._resolve_my_modal_name()

        # Fix C: Update the process_group (and num_partitions for Linear1D_Row)
        # cached inside every TP-sharded linear layer.  After a TP-degree change
        # the old group object may have been destroyed by Phase 8; these attrs
        # must point to the freshly created group.
        from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row

        new_tp = model.tp_group
        for m in model.module.modules():
            if isinstance(m, Linear1D_Col):
                m.process_group = new_tp
            if isinstance(m, Linear1D_Row):  # covers Linear1D_Row_ReduceScatter too
                m.process_group = new_tp
                m.num_partitions = dist.get_world_size(new_tp)

        # Update optimizer's process group references.
        if optimizer is not None:
            if hasattr(optimizer, "tp_pg"):
                optimizer.tp_pg = self.tp_group
            if hasattr(optimizer, "pp_pg"):
                optimizer.pp_pg = self.global_pp_group

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
                    encoder_tp_scatter=self._encoder_tp_scatter,
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
            module.config._attn_implementation = "bitfield_attention"
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
        sampler=None,
        **kwargs,
    ):
        """Prepare a dataloader for distributed training.

        Args:
            dataset: Dataset to load from.
            batch_size: Per data-parallel replica batch size. Used only when
                ``sampler`` is ``None`` (default ``DistributedSampler`` path).
            shuffle: Shuffle indices each epoch. Used only when ``sampler``
                is ``None``.
            seed: Random seed for shuffling and worker init.
            drop_last: Drop the last incomplete global batch. Used only when
                ``sampler`` is ``None``.
            pin_memory: Pin CPU memory in DataLoader workers.
            num_workers: Number of DataLoader worker processes.
            sampler: An optional pre-built batch sampler instance (e.g.
                ``SimpleSchedulerSampler`` or ``ManduSampler`` from MANDu).
                After ``init_distributed()`` the correct data-parallel
                ``rank`` and ``num_replicas`` are injected into the sampler
                (if it exposes those attributes), then the sampler is used
                as ``batch_sampler`` in the DataLoader.
                When ``None`` (default), a standard ``DistributedSampler``
                is created with the resolved DP rank / size.
            **kwargs: Extra arguments forwarded to ``DataLoader``.
        """
        assert dist.is_initialized(), "torch.distributed is not initialized."
        self.init_distributed()

        _kwargs = kwargs.copy()
        dp_size = self.pg_mesh.size(self.pg_mesh.dp_axis)
        dp_rank = self.pg_mesh.coords[0][self.pg_mesh.dp_axis]

        # Deterministic worker seeding
        def seed_worker(worker_id):
            worker_seed = seed
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            random.seed(worker_seed)

        if sampler is not None:
            # Inject the correct DP rank / size into the sampler so that
            # callers don't need to know process group internals upfront.
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

        # Default: standard DistributedSampler (no workload-aware reordering)
        default_sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            sampler=default_sampler,
            batch_size=batch_size,
            worker_init_fn=seed_worker,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **_kwargs,
        )

    def get_checkpoint_io(self) -> CheckpointIO:
        from cornstarch.plugin.multimodal_parallel_plugin.multimodal_parallel_checkpoint_io import (
            MultimodalParallelCheckpointIO,
        )

        return MultimodalParallelCheckpointIO(self)
