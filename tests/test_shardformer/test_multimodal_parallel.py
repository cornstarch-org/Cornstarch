from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from .model_zoo import (
    CLIPModelBase,
    LlamaForCausalLMBase,
    MistralForCausalLMBase,
    Phi3ForCausalLMBase,
    Phi4MultimodalAudioModelBase,
    Qwen2AudioEncoderBase,
    Qwen2ForCausalLMBase,
    Qwen2VisionTransformerBase,
    SiglipModelBase,
    WhisperEncoderBase,
)
from .utils import CornstarchMultimodalParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    siglip=SiglipModelBase,
    qwen2_vision=Qwen2VisionTransformerBase,
    # evaclip=EvaCLIPModelBase,
    # intern_vit=InternVisonModelBase,
)

audio_models = dict(
    qwen2_audio=Qwen2AudioEncoderBase,
    whisper=WhisperEncoderBase,
    phi4_audio=Phi4MultimodalAudioModelBase,
)

causal_lms = dict(
    # gemma2=Gemma2ForCausalLMBase,
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
        return 8

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
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "llm": language_pp_size},
        )


@instantiate_parametrized_tests
class VisionLanguageMultimodalContextParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 8

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, vision_sp_size, language_pp_size, language_sp_size",
        [
            (1, 1, 1, 1, 1),
            (1, 1, 2, 1, 2),
            (1, 2, 1, 2, 1),
            (1, 2, 2, 2, 2),
            (1, 2, 1, 1, 2),
            (2, 1, 1, 1, 1),
            (2, 1, 2, 1, 2),
            (2, 2, 1, 2, 1),
            (2, 2, 1, 1, 2),
        ],
        name_fn=lambda tp, vpp, vsp, lpp, lsp: f"tp={tp}, pp=({vpp},{lpp}), sp=({vsp},{lsp})",
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        tp_size: int,
        vision_pp_size: int,
        language_pp_size: int,
        vision_sp_size: int,
        language_sp_size: int,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            tp_size,
            {"vision": vision_pp_size, "llm": language_pp_size},
            {"vision": vision_sp_size, "llm": language_sp_size},
        )


@instantiate_parametrized_tests
class VisionAudioLanguageMultimodalParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 12

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize("audio_model_name", audio_models.keys(), lambda x: f"{x}")
    @parametrize(
        "tp_size, vision_pp_size, vision_sp_size, audio_pp_size, audio_sp_size, language_pp_size, language_sp_size",
        [
            (1, 1, 1, 1, 1, 1, 1),
            (1, 1, 2, 1, 2, 1, 2),
            (1, 2, 1, 2, 1, 2, 1),
            (1, 2, 2, 2, 2, 2, 2),
            (1, 2, 1, 2, 1, 1, 2),
            (2, 1, 1, 1, 1, 1, 1),
            (2, 1, 2, 1, 2, 1, 2),
            (2, 2, 1, 2, 1, 2, 1),
            (2, 2, 1, 2, 1, 1, 2),
        ],
        name_fn=lambda tp, vpp, vsp, app, asp, lpp, lsp: f"tp={tp}, pp=({vpp},{app},{lpp}), sp=({vsp},{asp},{lsp})",
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
        vision_sp_size: int,
        audio_sp_size: int,
        language_sp_size: int,
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
            {"vision": vision_sp_size, "audio": audio_sp_size, "llm": language_sp_size},
        )
