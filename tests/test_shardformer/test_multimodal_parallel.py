from typing import Any, Callable

from colossalai.booster import Booster
from colossalai.interface import OptimizerWrapper
from torch import dtype
from torch._C import dtype
from torch._tensor import Tensor
from torch.optim import Optimizer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.models.multimodal_language_model import MultimodalModel
from cornstarch.plugin.multimodal_parallel_plugin import MultimodalParallelModule

from .model_zoo import (
    CLIPModelBase,
    Gemma2ForCausalLMBase,
    LlamaForCausalLMBase,
    MistralForCausalLMBase,
    Phi3ForCausalLMBase,
    Qwen2AudioEncoderBase,
    Qwen2ForCausalLMBase,
    SiglipModelBase,
    WhisperEncoderBase,
)
from .utils import CornstarchMultimodalParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    siglip=SiglipModelBase,
    # dinov2=Dinov2ModelBase,
    # evaclip=EvaCLIPModelBase,
    # intern_vit=InternVisonModelBase,
)

audio_models = dict(
    qwen2_audio=Qwen2AudioEncoderBase,
    whisper=WhisperEncoderBase,
)

causal_lms = dict(
    gemma2=Gemma2ForCausalLMBase,
    llama=LlamaForCausalLMBase,
    mistral=MistralForCausalLMBase,
    phi3=Phi3ForCausalLMBase,
    qwen2=Qwen2ForCausalLMBase,
    # gemma=GemmaForCausalLMBase,
    # internlm2=InternLM2ForCausalLMBase,
    # mixtral=MixtralForCausalLMBase,
)


@instantiate_parametrized_tests
class VisionLanguageMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 16

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, language_pp_size",
        [
            (2, 1, 1),
            (2, 1, 3),
            (2, 2, 2),
        ],
        name_fn=lambda tp, vpp, lpp: f"tp={tp}, pp={vpp},{lpp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        fa: bool,
        precision: str,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "llm": language_pp_size},
            None,
            fa,
            precision,
        )


@instantiate_parametrized_tests
class VisionLanguageMultimodalContextParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self):
        return 12

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, language_pp_size",
        [
            (1, 1, 1),
            (2, 1, 1),
            (2, 2, 2),
        ],
        name_fn=lambda tp, vpp, lpp: f"tp={tp}, pp={vpp},{lpp}",
    )
    @parametrize("sp_mode", ["all_to_all", "ring_attn"], name_fn=lambda x: x)
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        sp_mode: str,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "llm": language_pp_size},
            sp_mode,
            True,
            "bf16",
        )


@instantiate_parametrized_tests
class VisionAudioLanguageMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 8

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, audio_pp_size, language_pp_size",
        [
            (2, 1, 1, 2),
            (1, 1, 1, 2),
        ],
        name_fn=lambda tp, vpp, app, lpp: f"tp={tp}, pp={vpp},{app},{lpp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        audio_pp_size: int,
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
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "audio": audio_pp_size, "llm": language_pp_size},
            None,
            fa,
            precision,
        )


@instantiate_parametrized_tests
class VisionAudioLanguageMultimodalParallelAnymask(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 12

    def postprocess_data_for_sharded_model(
        self, data: list[Tensor | dict[str, Tensor]], precision: dtype
    ) -> list | dict:
        data = super().postprocess_data_for_sharded_model(data, precision)
        assert isinstance(data, dict)
        del data["attention_mask"]
        data["input_ids"][:, 3:16] = 100
        data["labels"][:, 3:16] = -100
        data["input_ids"][:, 44:50] = 200
        data["labels"][:, 44:50] = -100

        return data

    def build_model_from_multimodal_plugin(
        self,
        tp_size: int,
        module_pp_size: dict[str, int],
        llm_sp_mode: str | None,
        test_config: dict[str, Any],
        precision: dtype,
    ) -> tuple[
        MultimodalModel,
        Optimizer,
        MultimodalParallelModule,
        OptimizerWrapper,
        Callable[..., Any],
        Booster,
    ]:
        (
            org_model,
            org_optimizer,
            shard_model,
            shard_optimizer,
            criterion,
            booster,
        ) = super().build_model_from_multimodal_plugin(
            tp_size, module_pp_size, llm_sp_mode, test_config, precision
        )
        module: MultimodalModel = shard_model.module
        module.set_token_ids(
            {
                "vision": 100,
                "audio": 200,
            }
        )

        return (
            org_model,
            org_optimizer,
            shard_model,
            shard_optimizer,
            criterion,
            booster,
        )

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, audio_pp_size, language_pp_size",
        [
            (2, 1, 1, 2),
            (1, 1, 1, 2),
        ],
        name_fn=lambda tp, vpp, app, lpp: f"tp={tp}, pp={vpp},{app},{lpp}",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        audio_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        audio_pp_size: int,
        language_pp_size: int,
    ):
        self.set_model(
            encoders={
                "vision": vision_models[vision_model_name](),
                "audio": audio_models[audio_model_name](),
            },
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "audio": audio_pp_size, "llm": language_pp_size},
            "ring_attn",
            False,
            "bf16",
            run_original_model=False,
            run_sharded_model=True,
        )
