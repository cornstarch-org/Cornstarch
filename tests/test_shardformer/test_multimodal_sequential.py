import copy
from contextlib import nullcontext
from typing import Any, Callable

import torch
from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from colossalai.lazy import LazyInitContext
from torch.optim import Adam, Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from transformers.modeling_utils import PreTrainedModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
)
from cornstarch.plugin.multimodal_sequential_plugin.multimodal_sequential_plugin import (
    EncoderCoalescedMultimodalParallelModule,
    EncoderCoalescedMultimodalParallelPlugin,
)

from .test_multimodal_parallel import audio_models, causal_lms, vision_models
from .utils import CornstarchMultimodalParallelBase


@instantiate_parametrized_tests
class EncoderCoalescedMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 12

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, encoder_pp_size, language_pp_size",
        [
            (2, 1, 2),
            (1, 1, 2),
            (2, 1, 3),
        ],
        name_fn=lambda tp, epp, lpp: f"tp={tp}, pp={epp},{lpp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
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

        self.run_coalesced_multimodal_parallel(
            tp_size,
            encoder_pp_size,
            language_pp_size,
            fa,
            precision,
        )

    def build_coalesced_model_from_multimodal_plugin(
        self,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
        test_config: dict[str, Any],
        precision: torch.dtype,
    ) -> tuple[
        MultimodalModel,
        Optimizer,
        EncoderCoalescedMultimodalParallelModule,
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
                        postprocess_module_callback=self.postprocess_callback,
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

        encoder_plugins: dict[str, ModalParallelPlugin] = {
            encoder_name: ModalParallelPlugin(
                tp_size=tp_size,
                pipeline_template=self.get_pipeline_template(
                    org_model.get_submodule(f"{encoder_name}_encoder"), encoder_pp_size
                ),
            )
            for encoder_name in ["vision", "audio"]
        }

        llm_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            pipeline_template=self.get_pipeline_template(
                org_model.language_model, language_pp_size
            ),
        )

        plugin = EncoderCoalescedMultimodalParallelPlugin(
            encoder_plugins=encoder_plugins,
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

    def run_coalesced_multimodal_parallel(
        self,
        tp_size: int,
        encoder_pp_size: int,
        language_pp_size: int,
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
        ) = self.build_coalesced_model_from_multimodal_plugin(
            tp_size=tp_size,
            encoder_pp_size=encoder_pp_size,
            language_pp_size=language_pp_size,
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
