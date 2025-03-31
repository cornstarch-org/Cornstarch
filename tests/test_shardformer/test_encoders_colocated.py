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
from transformers.modeling_outputs import ModelOutput

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)
from cornstarch.plugin.encoders_colocated_plugin.encoders_colocated_plugin import (
    EncodersColocatedMultimodalParallelModule,
    EncodersColocatedMultimodalParallelPlugin,
)
from cornstarch.plugin.encoders_colocated_plugin.encoders_colocated_stage_manager import (
    EncodersColocatedPipelineStageManager,
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


class EncodersColocatedMultimodalParallelBase(CornstarchMultimodalParallelBase):
    def parallelize_model(
        self,
        model: MultimodalModel,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
        encoder_sp_size: int,
        language_sp_size: int,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ):
        encoder_plugins: dict[str, ModalParallelPlugin] = {
            encoder_name: ModalParallelPlugin(
                tp_size=tp_size,
                sp_size=encoder_sp_size,
                sequence_parallelism_mode=(
                    "ring_attn" if encoder_sp_size > 1 else None
                ),
                pipeline_template=self.get_pipeline_template(
                    model.get_submodule(f"{encoder_name}_encoder"), encoder_pp_size
                ),
            )
            for encoder_name in ["vision", "audio"]
        }

        llm_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            sp_size=language_sp_size,
            sequence_parallelism_mode="ring_attn" if language_sp_size > 1 else None,
            pipeline_template=self.get_pipeline_template(
                model.language_model, language_pp_size
            ),
        )

        plugin = EncodersColocatedMultimodalParallelPlugin(
            encoders_plugins=encoder_plugins,
            language_model_plugin=llm_plugin,
            **test_config,
        )
        if precision == torch.bfloat16:
            model.to(dtype=precision)
            plugin.precision = None
        booster = Booster(plugin=plugin)

        optimizer = Adam(model.parameters(), lr=1e-3)
        model, optimizer, criterion, _, _ = booster.boost(
            model, optimizer, self.llm.loss_fn
        )

        return model, optimizer, criterion, booster

    def check_fn(
        self,
        booster: Booster,
        org_model: MultimodalModel,
        sharded_model: EncodersColocatedMultimodalParallelModule,
        org_optim: Optimizer,
        sharded_optim: OptimizerWrapper,
        org_output: ModelOutput,
        sharded_output: dict,
        org_loss: torch.Tensor,
        sharded_loss: torch.Tensor,
    ):
        my_modal_name = sharded_model.my_modal_name
        plugin: EncodersColocatedMultimodalParallelPlugin = booster.plugin
        stage_manager: EncodersColocatedPipelineStageManager = plugin.stage_manager

        # Loss check
        if stage_manager.is_last_stage(check_only_in_modal=False):
            check_loss(org_loss, sharded_loss, atol=self.llm.atol, rtol=self.llm.rtol)

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
                assert isinstance(my_modal_name, list)
                for encoder_name in my_modal_name:
                    modal_name = f"{encoder_name}_encoder"
                    org_encoder = org_model.get_submodule(modal_name).module
                    sharded_encoder = sharded_model.get_submodule(modal_name).module
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
                for encoder_name in my_modal_name:
                    modal_name = f"{encoder_name}_encoder"
                    org_encoder = org_model.get_submodule(modal_name).module
                    sharded_encoder = sharded_model.get_submodule(modal_name).module
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

        check_all_grad_tensors(grads_to_check)

    def build_colocated_model_from_multimodal_plugin(
        self,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
        encoder_sp_size: int,
        language_sp_size: int,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        MultimodalModel,
        Optimizer,
        EncodersColocatedMultimodalParallelModule,
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
            encoder_pp_size,
            language_pp_size,
            encoder_sp_size,
            language_sp_size,
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

    def run_colocated_multimodal_parallel(
        self,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
        encoder_sp_size: int = 1,
        language_sp_size: int = 1,
    ):
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
        ) = self.build_colocated_model_from_multimodal_plugin(
            tp_size=tp_size,
            encoder_pp_size=encoder_pp_size,
            language_pp_size=language_pp_size,
            encoder_sp_size=encoder_sp_size,
            language_sp_size=language_sp_size,
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


@instantiate_parametrized_tests
class EncodersColocatedMultimodalParallel(EncodersColocatedMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 6

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, encoder_pp_size, language_pp_size",
        [
            (1, 1, 1),
            (2, 1, 2),
            (1, 1, 2),
            (1, 2, 1),
            (2, 2, 1),
        ],
        name_fn=lambda tp, epp, lpp: f"tp={tp}, pp={epp},{lpp}",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
    ):
        self.set_model(
            encoders={
                "vision": vision_models[vision_model_name](),
                "audio": audio_models[audio_model_name](),
            },
            llm=causal_lms[language_model_name](),
        )

        self.run_colocated_multimodal_parallel(
            tp_size,
            encoder_pp_size,
            language_pp_size,
        )


@instantiate_parametrized_tests
class EncodersColocatedMultimodalContextParallel(
    EncodersColocatedMultimodalParallelBase
):
    @property
    def world_size(self) -> int:
        return 8

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, encoder_pp_size, encoder_sp_size, language_pp_size, language_sp_size",
        [
            (1, 1, 2, 1, 2),
            (1, 2, 1, 2, 1),
            (1, 2, 2, 2, 2),
            (1, 1, 2, 2, 1),
            (1, 2, 1, 1, 2),
            (2, 1, 2, 1, 2),
            (2, 2, 1, 2, 1),
            (2, 2, 1, 1, 2),
            (2, 1, 2, 2, 1),
        ],
        name_fn=lambda tp, epp, esp, lpp, lsp: f"tp={tp}, pp=({epp},{lpp}), sp=({esp},{lsp})",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        encoder_pp_size: int,
        encoder_sp_size: int,
        language_pp_size: int,
        language_sp_size: int,
    ):
        self.set_model(
            encoders={
                "vision": vision_models[vision_model_name](),
                "audio": audio_models[audio_model_name](),
            },
            llm=causal_lms[language_model_name](),
        )

        self.run_colocated_multimodal_parallel(
            tp_size,
            encoder_pp_size,
            language_pp_size,
            encoder_sp_size,
            language_sp_size,
        )
