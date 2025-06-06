import copy
import functools
import re
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.booster.plugin.hybrid_parallel_plugin import HybridParallelModule
from colossalai.interface import ModelWrapper, OptimizerWrapper
from colossalai.lazy import LazyInitContext
from colossalai.pipeline.schedule._utils import get_micro_batch
from colossalai.pipeline.stage_manager import PipelineStageManager
from colossalai.shardformer._utils import getattr_
from colossalai.shardformer.layer.qkv_fused_linear import (
    FusedLinear1D_Col,
    gather_fused_qkv_in_gpt2_style,
)
from colossalai.shardformer.policies.auto_policy import _fullname
from colossalai.tensor.d_tensor.api import (
    is_customized_distributed_tensor,
    is_distributed_tensor,
)
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.testing import assert_close
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    ModelOutput,
)
from transformers.modeling_utils import (
    ALL_ATTENTION_FUNCTIONS,
    PretrainedConfig,
    PreTrainedModel,
)

from cornstarch.kernel.interface import (
    bitfield_attention_forward,
    create_bitfield_attention_mask,
)
from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultimodalParallelPlugin,
    MultiModalPipelineStageManager,
)
from cornstarch.shardformer.policies.auto_policy import get_autopolicy
from cornstarch.shardformer.shard.shard_config import ContextParallelDistributionMode

from ..distributed_base import GlooDistributedTestBase

ALL_ATTENTION_FUNCTIONS.update({"bitfield_attention": bitfield_attention_forward})


class ModelClassBase(ABC):
    rtol, atol = 5e-3, 5e-3
    col_layers_to_check: list[str] = []
    row_layers_to_check: list[str] = []
    norm_layers_to_check: list[str] = []

    def __init__(self, model_class: PreTrainedModel, config: PretrainedConfig):
        self.model_class = model_class
        self.config = config

    @abstractmethod
    def loss_fn(
        self, x: ModelOutput, sp_group: dist.ProcessGroup = None
    ) -> torch.Tensor: ...

    @abstractmethod
    def data_gen_fn(self, num_batch: int) -> dict: ...

    def model_fn(self) -> PreTrainedModel:
        config = copy.deepcopy(self.config)
        config.pad_token_id = config.eos_token_id
        config._attn_implementation = "flash_attention_2"
        return self.model_class(config)


class ColossalaiHybridParallelBase(GlooDistributedTestBase):
    num_microbatches: int = 4
    microbatch_size: int = 2

    @property
    def world_size(self) -> int:
        return 8

    def set_model(self, model: ModelClassBase):
        self.model = model

    def postprocess_data_for_original_model(
        self, data: dict[str, torch.Tensor], precision: torch.dtype
    ) -> dict:
        assert isinstance(data, dict)

        new_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            new_data[k] = v.clone().to("cuda")

        return new_data

    def postprocess_data_for_sharded_model(
        self, data: dict[str, torch.Tensor], precision: torch.dtype
    ) -> dict:
        return self.postprocess_data_for_original_model(data, precision)

    def check_fn(
        self,
        booster: Booster,
        org_model: PreTrainedModel,
        sharded_model: ModelWrapper,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        plugin: HybridParallelPlugin = booster.plugin
        stage_manager = plugin.stage_manager
        # Loss check
        if stage_manager is None or stage_manager.is_last_stage():
            check_loss(
                org_loss, sharded_loss, atol=self.model.atol, rtol=self.model.rtol
            )

        dist.barrier()

        tp_group: dist.ProcessGroup = booster.plugin.tp_group
        sharded_model: PreTrainedModel = sharded_model.unwrap()

        # Gradient check
        grads_to_check = {}

        if stage_manager is None or stage_manager.is_first_stage():
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.row_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=0,
                    verbose=False,
                )
            )
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.col_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=1,
                    verbose=False,
                )
            )
            grads_to_check.update(
                get_grad_tensors_for_check(
                    org_model,
                    sharded_model,
                    self.model.norm_layers_to_check,
                    tp_group,
                    atol=self.model.atol,
                    rtol=self.model.rtol,
                    dim=1,
                    verbose=False,
                )
            )

        dist.barrier()

        # Update parameters
        org_optim.step()
        sharded_optim.step()

        # New parameter check
        if stage_manager is None or stage_manager.is_first_stage():
            check_weight(
                org_model,
                sharded_model,
                self.model.row_layers_to_check,
                tp_group,
                dim=0,
                atol=self.model.atol,
                rtol=self.model.rtol,
            )
            check_weight(
                org_model,
                sharded_model,
                self.model.col_layers_to_check,
                tp_group,
                dim=1,
                atol=self.model.atol,
                rtol=self.model.rtol,
            )

        dist.barrier()

        check_all_grad_tensors(grads_to_check)

    def run_hybrid_parallel(
        self,
        tp_size: int,
        pp_size: int,
        attention: str,
        precision: str,
        sp_mode: str | None = None,
        ring_attn_mode: ContextParallelDistributionMode | None = None,
    ):
        assert precision in ["bf16", "fp16"]
        precision = torch.bfloat16 if precision == "bf16" else torch.float16

        test_config = dict(
            tp_size=tp_size,
            pp_size=pp_size,
            zero_stage=0,
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
            attention=attention,
        )
        if sp_mode is not None:
            torch._dynamo.config.optimize_ddp = False
            test_config.update(
                {
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": sp_mode,
                    "sp_size": 2,
                    # context_parallel_distribution_mode will be set later,
                    # as HybirdParallelPlugin doesn't support it
                }
            )

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = self.build_model_from_hybrid_plugin(
            test_config=test_config, precision=precision
        )

        if sp_mode == "ring_attn":
            booster.plugin.shard_config.context_parallel_distribution_mode = (
                ring_attn_mode
            )

        try:
            org_model.gradient_checkpointing_enable({"use_reentrant": False})
            sharded_model.unwrap().gradient_checkpointing_enable(
                {"use_reentrant": False}
            )
        except ValueError:
            # Model does not support gradient checkpointing
            pass

        org_loss, org_output, sharded_loss, sharded_output = (
            self.run_forward_backward_with_hybrid_plugin(
                org_model=org_model,
                sharded_model=sharded_model,
                sharded_optimizer=sharded_optimizer,
                criterion=criterion,
                output_transform_fn=lambda x: x,
                booster=booster,
                precision=precision,
                attention=attention,
            )
        )

        self.check_fn(
            booster=booster,
            org_model=org_model,
            sharded_model=sharded_model,
            org_optim=org_optimizer,
            sharded_optim=sharded_optimizer,
            org_output=org_output,
            sharded_output=sharded_output,
            org_loss=org_loss,
            sharded_loss=sharded_loss,
        )

    def build_model_from_hybrid_plugin(
        self,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        PreTrainedModel,
        Optimizer,
        HybridParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        attention: str = test_config.pop("attention")
        test_config["enable_flash_attention"] = attention == "flash_attention_2"
        use_lazy_init: bool = test_config.pop("use_lazy_init", False)
        sp_mode: str = test_config.get("sequence_parallelism_mode", None)

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            org_model = self.model.model_fn().to(device="cuda")

            for param in org_model.parameters():
                if not param.isnan().any():
                    continue
                param.data = torch.randn_like(param.data)

            sharded_model = copy.deepcopy(org_model)
            sharded_model.config._attn_implementation = attention
        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)
        sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)

        with patch(
            "colossalai.shardformer.shard.sharder.get_autopolicy",
            return_value=get_autopolicy(_fullname(org_model)),
        ):
            plugin = HybridParallelPlugin(**test_config)
            if sp_mode == "ring_attn":
                # HybridParallelPlugin automatically enables flash attention for ring_attn. Disable it.
                plugin.enable_flash_attention = False
                plugin.shard_config.enable_flash_attention = False

            if precision == torch.bfloat16:
                org_model.to(dtype=precision)
                sharded_model.to(dtype=precision)
                plugin.precision = None
            else:
                # Do not cast org_model for fp16 here, as it will be casted in
                # torch.autocast amp
                plugin.precision = "fp16"
            booster = Booster(plugin=plugin)

            sharded_model, sharded_optimizer, criterion, _, _ = booster.boost(
                sharded_model, sharded_optimizer, self.model.loss_fn
            )

        return (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        )

    def run_forward_backward_with_hybrid_plugin(
        self,
        org_model: nn.Module,
        sharded_model: nn.Module,
        sharded_optimizer: Optimizer,
        criterion: Callable[[torch.Tensor, dist.ProcessGroup], torch.Tensor],
        output_transform_fn: Callable,
        booster: Booster,
        precision: torch.dtype,
        attention: str,
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs, sharded_model.sp_group)
            return loss

        data = self.model.data_gen_fn(self.microbatch_size * self.num_microbatches)

        unshard_test_data = self.postprocess_data_for_original_model(data, precision)
        shard_test_data = self.postprocess_data_for_sharded_model(data, precision)

        if attention == "bitfield_attention":
            shard_test_data["attention_mask"] = torch.full_like(
                shard_test_data["attention_mask"], (1 << 62) | 1, dtype=torch.int64
            )

        # use torch.autocast AMP for fp16 training test cases
        org_model.train()
        with torch.autocast(
            device_type="cuda", enabled=precision == torch.float16, dtype=torch.float16
        ):
            if isinstance(unshard_test_data, list):
                org_output = org_model(*unshard_test_data)
            else:
                org_output = org_model(**unshard_test_data)
            org_loss = criterion(org_output)
        org_loss.backward()

        sharded_model.train()
        if booster.plugin.stage_manager is not None:
            data_iter = iter([shard_test_data])
            sharded_output = booster.execute_pipeline(
                data_iter,
                sharded_model,
                _criterion,
                sharded_optimizer,
                return_loss=True,
                return_outputs=False,
            )
            sharded_loss = sharded_output["loss"]
        else:
            if isinstance(shard_test_data, list):
                sharded_output = sharded_model(*shard_test_data)
            else:
                sharded_output = sharded_model(**shard_test_data)
            sharded_loss = _criterion(sharded_output, shard_test_data)
            sharded_optimizer.backward(sharded_loss)

        return org_loss, org_output, sharded_loss, sharded_output


class CornstarchMultimodalParallelBase(GlooDistributedTestBase):
    num_microbatches: int = 6
    microbatch_size: int = 2
    token_ids: dict[str, int]

    def set_model(self, encoders: dict[str, ModelClassBase], llm: ModelClassBase):
        self.encoders = encoders
        self.llm = llm

    def check_fn(
        self,
        booster: Booster,
        org_model: MultimodalModel,
        sharded_model: MultimodalParallelModule,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        my_modal_name = sharded_model.my_modal_name
        plugin: MultimodalParallelPlugin = booster.plugin
        stage_manager: MultiModalPipelineStageManager = plugin.stage_manager

        # Loss check
        if stage_manager.is_last_stage(check_only_in_modal=False):
            check_loss(org_loss, sharded_loss, atol=self.llm.atol, rtol=self.llm.rtol)

        dist.barrier()

        tp_group: dist.ProcessGroup = plugin.tp_group
        sharded_model: MultimodalModel = sharded_model.unwrap()

        # Gradient check
        grads_to_check = {}

        if stage_manager.is_first_stage(check_only_in_modal=True):
            if my_modal_name == "language_model":
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_model.language_model,
                        sharded_model.language_model,
                        self.llm.row_layers_to_check,
                        tp_group,
                        atol=self.llm.atol,
                        rtol=self.llm.rtol,
                        dim=0,
                        verbose=False,
                    )
                )
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_model.language_model,
                        sharded_model.language_model,
                        self.llm.col_layers_to_check,
                        tp_group,
                        atol=self.llm.atol,
                        rtol=self.llm.rtol,
                        dim=1,
                        verbose=False,
                    )
                )
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_model.language_model,
                        sharded_model.language_model,
                        self.llm.norm_layers_to_check,
                        tp_group,
                        atol=self.llm.atol,
                        rtol=self.llm.rtol,
                        dim=1,
                        verbose=False,
                    )
                )
            else:
                encoder_name = my_modal_name.split("_")[0]
                org_encoder = org_model.get_submodule(my_modal_name).module
                sharded_encoder = sharded_model.get_submodule(my_modal_name).module
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_encoder,
                        sharded_encoder,
                        self.encoders[encoder_name].row_layers_to_check,
                        tp_group,
                        atol=self.encoders[encoder_name].atol,
                        rtol=self.encoders[encoder_name].rtol,
                        dim=0,
                        verbose=False,
                    )
                )
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_encoder,
                        sharded_encoder,
                        self.encoders[encoder_name].col_layers_to_check,
                        tp_group,
                        atol=self.encoders[encoder_name].atol,
                        rtol=self.encoders[encoder_name].rtol,
                        dim=1,
                        verbose=False,
                    )
                )
                grads_to_check.update(
                    get_grad_tensors_for_check(
                        org_encoder,
                        sharded_encoder,
                        self.encoders[encoder_name].norm_layers_to_check,
                        tp_group,
                        atol=self.encoders[encoder_name].atol,
                        rtol=self.encoders[encoder_name].rtol,
                        dim=1,
                        verbose=False,
                    )
                )

        dist.barrier()

        # Update parameters
        org_optim.step()
        sharded_optim.step()

        # New parameter check
        if stage_manager.is_first_stage(check_only_in_modal=True):
            if my_modal_name == "language_model":
                check_weight(
                    org_model.language_model,
                    sharded_model.language_model,
                    self.llm.row_layers_to_check,
                    tp_group,
                    dim=0,
                    atol=self.llm.atol,
                    rtol=self.llm.rtol,
                )
                check_weight(
                    org_model.language_model,
                    sharded_model.language_model,
                    self.llm.col_layers_to_check,
                    tp_group,
                    dim=1,
                    atol=self.llm.atol,
                    rtol=self.llm.rtol,
                )
            else:
                encoder_name = my_modal_name.split("_")[0]
                org_encoder = org_model.get_submodule(my_modal_name).module
                sharded_encoder = sharded_model.get_submodule(my_modal_name).module
                check_weight(
                    org_encoder,
                    sharded_encoder,
                    self.encoders[encoder_name].row_layers_to_check,
                    tp_group,
                    dim=0,
                    atol=self.encoders[encoder_name].atol,
                    rtol=self.encoders[encoder_name].rtol,
                )
                check_weight(
                    org_encoder,
                    sharded_encoder,
                    self.encoders[encoder_name].col_layers_to_check,
                    tp_group,
                    dim=1,
                    atol=self.encoders[encoder_name].atol,
                    rtol=self.encoders[encoder_name].rtol,
                )

        dist.barrier()

        check_all_grad_tensors(grads_to_check)

    def run_multimodal_parallel(
        self,
        tp_size: int,
        modal_pp_size: dict[str, int],
        modal_sp_size: dict[str, int] = {},
        run_original_model: bool = True,
        run_sharded_model: bool = True,
    ) -> tuple[nn.Module, ModelWrapper, Optimizer, OptimizerWrapper, Booster]:
        precision = torch.bfloat16

        test_config = dict(
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
        )

        (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        ) = self.build_model_from_multimodal_plugin(
            tp_size=tp_size,
            module_pp_size=modal_pp_size,
            module_sp_size=modal_sp_size,
            test_config=test_config,
            precision=precision,
        )

        org_loss, org_output, sharded_loss, sharded_output = (
            self.run_forward_backward_with_multimodal_plugin(
                org_model=org_model,
                sharded_model=sharded_model,
                sharded_optimizer=sharded_optimizer,
                criterion=criterion,
                output_transform_fn=lambda x: x,
                booster=booster,
                precision=precision,
                run_original_model=run_original_model,
                run_sharded_model=run_sharded_model,
            )
        )

        # checking correctness can only be done when running both models
        if run_original_model and run_sharded_model:
            self.check_fn(
                booster=booster,
                org_model=org_model,
                sharded_model=sharded_model,
                org_optim=org_optimizer,
                sharded_optim=sharded_optimizer,
                org_output=org_output,
                sharded_output=sharded_output,
                org_loss=org_loss,
                sharded_loss=sharded_loss,
            )

        return org_model, sharded_model, org_optimizer, sharded_optimizer, booster

    """
    AS multiple encoders may use `hidden_states`,
    we cannot put `hidden_states` in the input.
    Instead, we change input name to `hidden_states` in preprocess callback,
    which only includes encoder-specific inputs and no possibility of duplicates.
    """

    @staticmethod
    def phi4_audio_preprocess_callback(inputs: dict) -> dict:
        inputs = {
            "hidden_states": inputs["audio_input_features"],
            "mask": inputs["audio_attention_mask"],
        }

        return inputs

    @staticmethod
    def qwen2_vision_preprocess_callback(inputs: dict) -> dict:
        inputs = {
            "hidden_states": inputs["pixel_values"].view(
                -1, inputs["pixel_values"].shape[-1]
            ),
            "grid_thw": inputs["image_grid_thw"].view(
                -1, inputs["image_grid_thw"].shape[-1]
            ),
        }

        return inputs

    @staticmethod
    def qwen2_vision_postprocess_projector_callback(
        inputs: dict, output: ModelOutput, language_hidden_size: int
    ) -> ModelOutput:
        # Qwen2Vision specific
        batch_size = inputs[
            "image_grid_thw" if "image_grid_thw" in inputs else "grid_thw"
        ].shape[0]
        output.hidden_states = output.hidden_states.view(
            batch_size, -1, language_hidden_size
        )

        output.hidden_states = output.hidden_states[:, :32]
        return output

    @staticmethod
    def postprocess_projector_callback(
        inputs: dict, output: ModelOutput
    ) -> ModelOutput:
        output.hidden_states = output.hidden_states[:, :32]
        return output

    def build_model_from_pretrained(
        self,
        encoder_paths: dict[str, tuple[Path, Path]],
        language_model_path: Path,
    ) -> MultimodalModel:
        encoders: dict[str, ModalEncoderModule] = {}
        for modal_key, (module_path, projector_path) in encoder_paths.items():
            if module_path is not None:
                encoder = self.encoders[modal_key].model_class.from_pretrained(
                    module_path,
                    torch_dtype=torch.bfloat16,
                    _attn_implementation="eager",
                )
            else:
                encoder = self.encoders[modal_key].model_fn()

            if projector_path is not None:
                projector = MultimodalProjector.from_pretrained(
                    projector_path, torch_dtype=torch.bfloat16
                )
            else:
                projector = None
            encoders[modal_key] = ModalEncoderModule(encoder, projector)

        if language_model_path is not None:
            llm = self.llm.model_class.from_pretrained(
                language_model_path, torch_dtype=torch.bfloat16
            )
        else:
            llm = self.llm.model_fn()

        model = MultimodalModel(
            encoders=encoders,
            language_model=llm,
        ).to(dtype=torch.bfloat16)

        num_existing_tokens = llm.get_input_embeddings().weight.shape[0]
        self.token_ids = {
            encoder_name: num_existing_tokens + encoder_index
            for encoder_index, encoder_name in enumerate(encoders)
        }
        model.set_modality_token_ids(self.token_ids)
        model.gradient_checkpointing_enable({"use_reentrant": False})

        return model

    def build_model_from_config(self) -> MultimodalModel:
        encoders: dict[str, ModalEncoderModule] = {}
        for modal_key, model_base in self.encoders.items():
            encoder = model_base.model_fn()

            if model_base.__class__.__name__ == "Qwen2VisionTransformerBase":
                encoders[modal_key] = ModalEncoderModule(
                    encoder,
                    preprocess_callback=self.qwen2_vision_preprocess_callback,
                    postprocess_projector_callback=functools.partial(
                        self.qwen2_vision_postprocess_projector_callback,
                        language_hidden_size=self.llm.config.hidden_size,
                    ),
                    additional_args=["pixel_values", "image_grid_thw"],
                )
            elif model_base.__class__.__name__ == "Phi4MultimodalAudioModelBase":
                encoders[modal_key] = ModalEncoderModule(
                    encoder,
                    preprocess_callback=self.phi4_audio_preprocess_callback,
                    postprocess_projector_callback=self.postprocess_projector_callback,
                    additional_args=[
                        "audio_input_features",
                        "audio_attention_mask",
                        "chunk_pad_size",
                    ],
                )
            else:
                encoders[modal_key] = ModalEncoderModule(
                    encoder,
                    postprocess_projector_callback=self.postprocess_projector_callback,
                )

        llm = self.llm.model_fn()

        model = MultimodalModel(
            encoders=encoders,
            language_model=llm,
        ).to(dtype=torch.bfloat16, device="cuda")

        num_existing_tokens = llm.get_input_embeddings().weight.shape[0]
        self.token_ids = {
            encoder_name: num_existing_tokens + encoder_index
            for encoder_index, encoder_name in enumerate(encoders)
        }
        model.set_modality_token_ids(token_ids=self.token_ids)
        model.gradient_checkpointing_enable({"use_reentrant": False})

        return model

    def parallelize_model(
        self,
        model: MultimodalModel,
        tp_size: int,
        module_pp_size: dict[str, int],
        module_sp_size: dict[str, int],
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        MultimodalParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        plugins: dict[str, ModalParallelPlugin] = {}
        for modal_name, pp_size in module_pp_size.items():
            if modal_name == "llm":
                llm_sp_size = module_sp_size.get("llm", 1)
                llm_plugin = ModalParallelPlugin(
                    tp_size=tp_size,
                    sp_size=llm_sp_size,
                    sequence_parallelism_mode="ring_attn" if llm_sp_size > 1 else None,
                    pipeline_template=self.get_pipeline_template(
                        model.language_model, pp_size
                    ),
                )
            else:
                modal_sp_size = module_sp_size.get(modal_name, 1)
                plugins[modal_name] = ModalParallelPlugin(
                    tp_size=tp_size,
                    sp_size=modal_sp_size,
                    sequence_parallelism_mode=(
                        "ring_attn" if modal_sp_size > 1 else None
                    ),
                    pipeline_template=self.get_pipeline_template(
                        model.get_submodule(f"{modal_name}_encoder"), pp_size
                    ),
                )

        plugin = MultimodalParallelPlugin(
            encoder_plugins=plugins,
            language_model_plugin=llm_plugin,
            **test_config,
        )
        if precision == torch.bfloat16:
            model.to(dtype=precision)
            plugin.precision = None
        else:
            # Do not cast org_model for fp16 here, as it will be casted in
            # torch.autocast amp
            plugin.precision = "fp16"
        booster = Booster(plugin=plugin)

        optimizer = Adam(model.parameters(), lr=1e-3)
        model, optimizer, criterion, _, _ = booster.boost(
            model, optimizer, self.llm.loss_fn
        )

        return model, optimizer, criterion, booster

    def build_model_from_multimodal_plugin(
        self,
        tp_size: int,
        module_pp_size: dict[str, int],
        module_sp_size: dict[str, int],
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        MultimodalModel,
        Optimizer,
        MultimodalParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        use_lazy_init: bool = test_config.pop("use_lazy_init", False)

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            org_model = self.build_model_from_config()
            sharded_model = copy.deepcopy(org_model)

        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)

        sharded_model, sharded_optimizer, criterion, booster = self.parallelize_model(
            sharded_model,
            tp_size,
            module_pp_size,
            module_sp_size,
            test_config,
            precision,
        )

        org_model.update_language_model_to_use_bitfield_attention_mask()
        sharded_model.unwrap().update_language_model_to_use_bitfield_attention_mask()

        return (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        )

    @staticmethod
    def get_pipeline_template(model: nn.Module, num_stages: int) -> PipelineTemplate:
        modules = PipelineTemplate.get_modules(model)
        num_layers = sum(bool(re.search(r"\.\d", s)) for s in modules)

        # Get the number of layers per stage
        base_size = num_layers // num_stages
        remainder = num_layers % num_stages
        num_layers_per_stage = [
            base_size + 1 if i < remainder else base_size for i in range(num_stages)
        ]
        assert sum(num_layers_per_stage) == num_layers

        first_layer_index = next(
            i for i, layer in enumerate(modules) if re.search(r"\.0", layer)
        )
        last_layer_index = next(
            i
            for i, layer in enumerate(modules)
            if re.search(rf"\.{num_layers - 1}", layer)
        )

        modules_per_stages = [[] for _ in range(num_stages)]
        modules_per_stages[0].extend(modules[:first_layer_index])
        layer_idx = 0
        for stage_idx, num_layers in enumerate(num_layers_per_stage):
            idx = first_layer_index + layer_idx
            modules_per_stages[stage_idx].extend(modules[idx : idx + num_layers])
            layer_idx += num_layers
        modules_per_stages[-1].extend(modules[last_layer_index + 1 :])

        return PipelineTemplate(
            (
                model.config[0].model_type
                if isinstance(model, ModalEncoderModule)
                else model.config.model_type
            ),
            modules_per_stages,
        )

    def postprocess_data_for_original_model(
        self, data: dict[str, torch.Tensor], precision: torch.dtype
    ) -> dict:
        assert isinstance(data, dict)

        new_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            new_data[k] = v.clone().to("cuda")

        """
        Inject encoder tokens to the input_ids for the multimodal model.
        """
        input_ids: torch.Tensor = new_data["input_ids"]
        encoder_tokens: list[torch.Tensor] = []
        for modal_key in self.encoders.keys():
            # num_encoder_tokens is a list[int] type, a list of number of tokens for each batch.
            # Implement a 2D tensor with the shape of (batch_size, num_encoder_tokens)
            encoder_tokens.append(
                torch.full(
                    (input_ids.shape[0], 32),
                    fill_value=self.token_ids[modal_key],
                    dtype=torch.long,
                    device=input_ids.device,
                )
            )

        # prepend it to input_ids
        input_ids = torch.cat(encoder_tokens + [input_ids], dim=1)
        new_data["input_ids"] = input_ids
        new_data["labels"] = input_ids
        new_data["use_cache"] = False

        return new_data

    def postprocess_data_for_sharded_model(
        self, data: dict[str, torch.Tensor], precision: torch.dtype
    ) -> dict:
        return self.postprocess_data_for_original_model(data, precision)

    def run_forward_backward_with_multimodal_plugin(
        self,
        org_model: nn.Module,
        sharded_model: nn.Module,
        sharded_optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        output_transform_fn: Callable,
        booster: Booster,
        precision: torch.dtype,
        run_original_model: bool = True,
        run_sharded_model: bool = True,
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = {}
        batch_size = self.microbatch_size * self.num_microbatches
        for model_base in self.encoders.values():
            data.update(model_base.data_gen_fn(batch_size))
        data.update(self.llm.data_gen_fn(batch_size))

        unshard_test_data = self.postprocess_data_for_original_model(data, precision)
        shard_test_data = self.postprocess_data_for_sharded_model(data, precision)

        org_loss, org_output = None, None
        if run_original_model:
            org_model.train()

            # org_output = org_model(**unshard_test_data)
            # org_loss = criterion(org_output)
            # org_loss.backward()

            org_loss = torch.scalar_tensor(0, device="cuda")
            for i in range(self.num_microbatches):
                input = get_micro_batch(
                    unshard_test_data, i * self.microbatch_size, self.microbatch_size
                )
                for k, v in input.items():
                    if isinstance(v, torch.Tensor):
                        input[k] = v.contiguous()
                output = org_model(**input)
                loss = criterion(output) / self.num_microbatches
                loss.backward()
                org_loss.add_(loss.data)

        sharded_loss, sharded_output = None, None
        if run_sharded_model:
            sharded_model.train()
            data_iter = iter([shard_test_data])
            sharded_output = booster.execute_pipeline(
                data_iter,
                sharded_model,
                _criterion,
                sharded_optimizer,
                return_loss=True,
                return_outputs=False,
            )
            sharded_loss = sharded_output["loss"]

        return org_loss, org_output, sharded_loss, sharded_output


def check_output_hidden_state(
    org_output: Tensor,
    sharded_output: Tensor,
    stage_manager: Optional[PipelineStageManager] = None,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    assert (
        stage_manager is None or stage_manager.is_last_stage()
    ), "check_output_hidden_state only supports last stage"
    org_hidden_state = org_output.last_hidden_state
    sharded_hidden_state = sharded_output.last_hidden_state

    assert_close(
        org_hidden_state.float(), sharded_hidden_state.float(), atol=atol, rtol=rtol
    )


def check_loss(
    org_loss: Tensor,
    sharded_loss: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-3,
):
    assert_close(org_loss.float(), sharded_loss.float(), atol=atol, rtol=rtol)


def check_weight(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: Optional[dist.ProcessGroup] = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_weight = getattr_(org_model, suffix).weight
        sharded_weight = getattr_(sharded_model, suffix).weight

        # skip if layer is not held by this process
        if sharded_weight is None:
            continue

        if is_distributed_tensor(sharded_weight) or is_customized_distributed_tensor(
            sharded_weight
        ):
            sharded_module = getattr_(sharded_model, suffix)
            if isinstance(sharded_module, FusedLinear1D_Col):
                sharded_weight = gather_fused_qkv_in_gpt2_style(
                    sharded_weight, sharded_module.split_sizes, tp_group
                )
            else:
                sharded_weight_list = [
                    torch.zeros_like(sharded_weight, device="cuda")
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(sharded_weight_list, sharded_weight, tp_group)
                sharded_weight = torch.cat(sharded_weight_list, dim=dim)

        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' weight: {org_weight}, {sharded_weight}")

        assert_close(org_weight.float(), sharded_weight.float(), atol=atol, rtol=rtol)


def get_grad_tensors_for_check(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: dist.ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
    name: str = None,
):
    grad_to_check = {}
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(
            shard_weight
        ):
            sharded_module = getattr_(sharded_model, suffix)
            if isinstance(sharded_module, FusedLinear1D_Col):
                shard_grad = gather_fused_qkv_in_gpt2_style(
                    shard_grad, sharded_module.split_sizes, tp_group
                )
            else:
                shard_grad_list = [
                    torch.zeros_like(shard_grad, device="cuda")
                    for _ in range(dist.get_world_size(tp_group))
                ]
                dist.all_gather(shard_grad_list, shard_grad, tp_group)
                shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        grad_to_check[suffix] = {
            "org_grad": org_grad.float(),
            "shard_grad": shard_grad.float(),
            "rtol": rtol,
            "atol": atol,
        }

    return grad_to_check


# used by sam/blip2
def check_grad(
    org_model: nn.Module,
    sharded_model: nn.Module,
    layer_suffix: list[str],
    tp_group: dist.ProcessGroup = None,
    dim: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    verbose: bool = False,
):
    for suffix in layer_suffix:
        org_grad = getattr_(org_model, suffix).weight.grad
        shard_grad = getattr_(sharded_model, suffix).weight.grad
        shard_weight = getattr_(sharded_model, suffix).weight
        if is_distributed_tensor(shard_weight) or is_customized_distributed_tensor(
            shard_weight
        ):
            shard_grad_list = [
                torch.zeros_like(shard_grad, device="cuda")
                for _ in range(dist.get_world_size(tp_group))
            ]
            dist.all_gather(shard_grad_list, shard_grad, tp_group)
            shard_grad = torch.cat(shard_grad_list, dim=dim)

        # embedding may be resized when using tensor parallel
        if shard_grad.shape[0] > org_grad.shape[0]:
            shard_grad = shard_grad[: org_grad.shape[0], :]
        if verbose and dist.get_rank() == 0:
            print(f"'{suffix}' grad: {org_grad}, {shard_grad}")

        assert_close(org_grad.float(), shard_grad.float(), rtol=rtol, atol=atol)


def unwrap_model(
    module: nn.Module,
    base_model_class_name: Optional[str] = None,
    base_model_attribute_name: Optional[str] = None,
):
    if isinstance(module, HybridParallelModule):
        module = module.unwrap()
    if base_model_class_name is None:
        return module
    if module.__class__.__name__ == base_model_class_name:
        return module
    return getattr(module, base_model_attribute_name, None)


def check_all_grad_tensors(check_tensors):
    """
    "org_grad": tensor to be compared from the original model
    "shard_grad": tensor to be compared from the sharded model
    """
    for suffix, check_info in check_tensors.items():
        org_grad = check_info["org_grad"]
        shard_grad = check_info["shard_grad"]
        rtol = check_info["rtol"]
        atol = check_info["atol"]
        assert_close(org_grad, shard_grad, atol=atol, rtol=rtol)
