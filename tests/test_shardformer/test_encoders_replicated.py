import copy
from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from colossalai.lazy import LazyInitContext
from torch.optim import Adam, Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.encoders_replicated_plugin.encoders_replicated_plugin import (
    EncodersReplicatedMultimodalParallelModule,
    EncodersReplicatedMultimodalParallelPlugin,
)
from cornstarch.plugin.encoders_replicated_plugin.encoders_replicated_stage_manager import (
    EncodersReplicatedPipelineStageManager,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
)

from .test_multimodal_parallel import audio_models, causal_lms, vision_models
from .utils import (
    CornstarchMultimodalParallelBase,
    check_all_grad_tensors,
    check_loss,
    check_weight,
    get_grad_tensors_for_check,
)


@instantiate_parametrized_tests
class EncoderReplicatedMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 8

    def check_fn(
        self,
        booster: Booster,
        org_model: MultimodalModel,
        sharded_model: EncodersReplicatedMultimodalParallelModule,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        plugin: EncodersReplicatedMultimodalParallelPlugin = booster.plugin
        stage_manager: EncodersReplicatedPipelineStageManager = plugin.stage_manager

        # Loss check
        if stage_manager.is_last_stage(check_only_in_modal=False):
            check_loss(org_loss, sharded_loss, atol=self.llm.atol, rtol=self.llm.rtol)

        tp_group: dist.ProcessGroup = plugin.tp_group
        sharded_model: MultimodalModel = sharded_model.unwrap()

        # Gradient check
        grads_to_check = {}

        if stage_manager.is_first_stage(check_only_in_modal=True):
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

        # Update parameters
        org_optim.step()
        sharded_optim.step()

        # New parameter check
        if stage_manager.is_first_stage(check_only_in_modal=True):
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

        check_all_grad_tensors(grads_to_check)

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, pp_size",
        [
            (2, 2),
            (1, 2),
            (1, 4),
        ],
        name_fn=lambda tp, pp: f"tp={tp}, pp={pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        pp_size: int,
        fa: bool,
        precision: str,
    ):
        self.set_model(
            encoders={
                "vision": vision_models[vision_model_name](),
                "audio": audio_models[audio_model_name](),
            },
            llm=causal_lms[language_model_name](),
        )

        self.run_replicated_multimodal_parallel(
            tp_size,
            pp_size,
            fa,
            precision,
        )

    def postprocess_projector_callback(
        self,
        inputs: dict,
        output: BaseModelOutput | tuple,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pad_token_id: int = -1,
        ignore_index: int = -100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Does not add encoder output to LLM input.
        return inputs_embeds, attention_mask, inputs.get("position_ids", None), labels

    def build_replicated_model_from_multimodal_plugin(
        self,
        tp_size: int,
        pp_size: int,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        MultimodalModel,
        Optimizer,
        EncodersReplicatedMultimodalParallelModule,
        OptimizerWrapper,
        Callable,
        Booster,
    ]:
        use_lazy_init: bool = test_config.pop("use_lazy_init", False)
        use_flash_attention: bool = test_config["enable_flash_attention"]

        ctx = LazyInitContext() if use_lazy_init else nullcontext()
        with ctx:
            encoders: dict[str, PreTrainedModel] = {}

            for encoder_name, model_base in self.encoders.items():
                encoders[encoder_name] = model_base.model_fn(use_flash_attention)

            llm = self.llm.model_fn(use_flash_attention)

            org_model = MultimodalModel(
                encoders={
                    encoder_name: ModalEncoderModule(
                        module,
                        postprocess_projector_callback=self.postprocess_projector_callback,
                    )
                    for encoder_name, module in encoders.items()
                },
                language_model=llm,
            ).to("cuda")
            sharded_model = copy.deepcopy(org_model)
        if use_lazy_init:
            ctx.materialize(org_model)

        org_optimizer = Adam(org_model.parameters(), lr=1e-3)
        sharded_optimizer = Adam(sharded_model.parameters(), lr=1e-3)

        llm_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            pipeline_template=self.get_pipeline_template(
                org_model.language_model, pp_size
            ),
        )

        plugin = EncodersReplicatedMultimodalParallelPlugin(
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
            sharded_model, sharded_optimizer, self.llm.loss_fn
        )

        return (
            org_model,
            org_optimizer,
            sharded_model,
            sharded_optimizer,
            criterion,
            booster,
        )

    def run_replicated_multimodal_parallel(
        self,
        tp_size: int,
        pp_size: int,
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
        ) = self.build_replicated_model_from_multimodal_plugin(
            tp_size=tp_size,
            pp_size=pp_size,
            test_config=test_config,
            precision=precision,
        )

        org_model.gradient_checkpointing_enable()
        sharded_model.unwrap().gradient_checkpointing_enable()

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
