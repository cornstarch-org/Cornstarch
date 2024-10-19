import copy
import re
from contextlib import nullcontext
from typing import Any, Callable
from unittest.mock import patch

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from colossalai.lazy import LazyInitContext
from torch import nn
from torch.optim import Adam, Optimizer
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.clip import CLIPVisionConfig
from transformers.models.dinov2 import Dinov2Config
from transformers.models.gemma import GemmaConfig
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.llama import LlamaConfig
from transformers.models.mistral import MistralConfig
from transformers.models.mixtral import MixtralConfig
from transformers.models.phi3 import Phi3Config
from transformers.models.qwen2 import Qwen2Config
from transformers.models.siglip import SiglipVisionConfig

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)

from ..utils import PolicyTestBase


class CornstarchMultimodalParallelBase(PolicyTestBase):
    model_class: dict[str, PreTrainedModel]
    config: dict[str, PretrainedConfig]

    @staticmethod
    def loss_fn(outputs: CausalLMOutputWithPast) -> torch.Tensor:
        return outputs.loss

    def model_fn(self, fa: bool) -> MultimodalModel:
        config = copy.deepcopy(self.config)
        modals: dict[str, PreTrainedModel] = {}

        for modal_name, modal_config in config.items():
            modal_config.pad_token_id = modal_config.eos_token_id
            modal_config._attn_implementation = "flash_attention_2" if fa else "eager"
            modal_model = self.model_class[modal_name](modal_config)
            modals[modal_name] = modal_model

        llm = modals.pop("llm")

        return MultimodalModel(
            encoders={
                modal_name: ModalEncoderModule(modal)
                for modal_name, modal in modals.items()
            },
            language_model=llm,
        )

    def run_multimodal_parallel(
        self,
        tp_size: int,
        modal_pp_size: dict[str, int],
        llm_sp_mode: str | None,
        fa: bool,
        precision: str,
    ):
        assert precision in ["bf16", "fp16"]
        precision = torch.bfloat16 if precision == "bf16" else torch.float16

        test_config = dict(
            num_microbatches=self.num_microbatches,
            microbatch_size=self.microbatch_size,
            initial_scale=1,
            enable_flash_attention=fa,
            precision=precision,
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
            modal_pp_size=modal_pp_size,
            llm_sp_mode=llm_sp_mode,
            test_config=test_config,
            precision=precision,
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

        with (
            patch.object(
                dist,
                "batch_isend_irecv",
                new=PolicyTestBase.batch_isend_irecv_gloo,
            ),
            patch.object(
                dist,
                "all_to_all",
                new=PolicyTestBase.all_to_all_gloo,
            ),
            patch.object(
                dist,
                "all_to_all_single",
                new=PolicyTestBase.all_to_all_single_gloo,
            ),
            patch(
                "colossalai.pipeline.p2p._check_device",
                return_value=(torch.device("cuda"), False),
            ),
        ):
            org_loss, org_output, sharded_loss, sharded_output = (
                self.run_forward_backward_with_multimodal_plugin(
                    org_model=org_model,
                    sharded_model=sharded_model,
                    sharded_optimizer=sharded_optimizer,
                    criterion=criterion,
                    output_transform_fn=lambda x: x,
                    booster=booster,
                    precision=precision,
                )
            )

        self.check_fn(
            booster,
            org_model,
            sharded_model,
            org_optimizer,
            sharded_optimizer,
            org_output,
            sharded_output,
            org_loss,
            sharded_loss,
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
            i for i, layer in enumerate(modules) if re.search("\.0", layer)
        )
        last_layer_index = next(
            i
            for i, layer in enumerate(modules)
            if re.search(f"\.{num_layers - 1}", layer)
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

    def build_model_from_multimodal_plugin(
        self,
        tp_size: int,
        modal_pp_size: dict[str, int],
        llm_sp_mode: str | None,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        PreTrainedModel,
        Optimizer,
        MultimodalParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        use_lazy_init: bool = test_config.pop("use_lazy_init", False)
        use_flash_attention: bool = test_config["enable_flash_attention"]

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            org_model = self.model_fn(use_flash_attention).to(device="cuda")
            sharded_model = copy.deepcopy(org_model)
        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)
        sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)

        plugins: dict[str, ModalParallelPlugin] = {}
        for modal_name, pp_size in modal_pp_size.items():
            if modal_name == "llm":
                llm_plugin = ModalParallelPlugin(
                    tp_size=tp_size,
                    pipeline_template=self.get_pipeline_template(
                        org_model.language_model, pp_size
                    ),
                )
            else:
                plugins[modal_name] = ModalParallelPlugin(
                    tp_size=tp_size,
                    pipeline_template=self.get_pipeline_template(
                        org_model.get_submodule(f"{modal_name}_encoder"), pp_size
                    ),
                )

        plugin = MultimodalParallelPlugin(
            encoder_plugins=plugins,
            language_model_plugin=llm_plugin,
            **test_config,
        )
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
            sharded_model, sharded_optimizer, self.loss_fn
        )

        return (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        )

    def run_forward_backward_with_multimodal_plugin(
        self,
        org_model: nn.Module,
        sharded_model: nn.Module,
        sharded_optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        output_transform_fn: Callable,
        booster: Booster,
        precision: torch.dtype,
    ):
        def _criterion(outputs: BaseModelOutputWithPast, inputs: Any):
            outputs = output_transform_fn(outputs)
            loss = criterion(outputs)
            return loss

        data = self.data_gen_fn()

        shard_test_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            shard_test_data[k] = v.clone().to("cuda")
        unshard_test_data = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v = v.to(dtype=precision)
            unshard_test_data[k] = v.clone().to("cuda")

        org_model.train()
        with torch.autocast(
            device_type="cuda", enabled=precision == torch.float16, dtype=torch.float16
        ):
            org_output = org_model(**unshard_test_data)
            org_loss = criterion(org_output).to(precision)
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
            sharded_output = sharded_model(**shard_test_data)
            sharded_loss = criterion(sharded_output).to(precision)
            sharded_optimizer.backward(sharded_loss)

        return org_loss, org_output, sharded_loss, sharded_output


llama_config = LlamaConfig()
gemma_config = GemmaConfig()
gemma2_config = Gemma2Config()
mistral_config = MistralConfig()
mixtral_config = MixtralConfig()
phi3_config = Phi3Config()
qwen2_config = Qwen2Config()

for language_config in [
    llama_config,
    gemma_config,
    gemma2_config,
    mistral_config,
    mixtral_config,
    phi3_config,
    qwen2_config,
]:
    language_config.hidden_size = 256
    language_config.intermediate_size = 256
    language_config.num_attention_heads = 16
    language_config.num_key_value_heads = 16
    language_config.num_hidden_layers = 4
    language_config.use_cache = False
    language_config._attn_implementation = "eager"
    # TODO: Gemma uses tie_word_embeddings True, in which case the tests fail.
    # Implement automatic gradient synchronization between tied weights.
    # Existing explicit synchronization is not enough as there are encoders
    # that need to have gradients propagated "after" the weights are synchronized.
    language_config.tie_word_embeddings = False

# GQA adjustment. Models not in this list use MHA.
for language_config in [
    gemma2_config,
    mistral_config,
    mixtral_config,
    qwen2_config,
]:
    language_config.num_key_value_heads = 8

# MoE adjustment
for language_config in [mixtral_config]:
    language_config.num_local_experts = 4
    language_config.num_experts_per_tok = 1

clip_config = CLIPVisionConfig()
siglip_config = SiglipVisionConfig()
dinov2_config = Dinov2Config()

for vision_config in [clip_config, siglip_config, dinov2_config]:
    vision_config.hidden_size = 256
    vision_config.intermediate_size = 256
    vision_config.num_attention_heads = 8
    vision_config.num_hidden_layers = 4
    vision_config._attn_implementation = "eager"
    vision_config.use_cache = False
