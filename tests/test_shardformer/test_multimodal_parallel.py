import pytest
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

    def postprocess_data_for_original_model(self, data, precision):
        return super().postprocess_data_for_original_model(data, precision)

    def postprocess_data_for_sharded_model(self, data, precision):
        return self.postprocess_data_for_original_model(data, precision)

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize(
        "vtp, vpp, ltp, lpp",
        [
            (1, 1, 1, 1),
            (1, 2, 2, 1),
            (1, 2, 1, 2),
            (2, 1, 2, 1),
            (2, 1, 1, 2),
            (2, 2, 2, 2),
        ],
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        vtp: int,
        ltp: int,
        vpp: int,
        lpp: int,
    ):
        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            {"vision": vtp, "llm": ltp},
            {"vision": vpp, "llm": lpp},
        )


@instantiate_parametrized_tests
class VisionLanguageMultimodalContextParallel(CornstarchMultimodalParallelBase):
    @property
    def world_size(self) -> int:
        return 8

    def postprocess_data_for_original_model(self, data, precision):
        return super().postprocess_data_for_original_model(data, precision)

    def postprocess_data_for_sharded_model(self, data, precision):
        return self.postprocess_data_for_original_model(data, precision)

    @parametrize("vision_model_name", vision_models.keys(), lambda x: f"{x}")
    @parametrize("language_model_name", causal_lms.keys(), lambda x: f"{x}")
    @parametrize(
        "vtp, vpp, vsp, ltp, lpp, lsp",
        [
            (1, 2, 1, 1, 1, 2),  # 1tp+1sp -> 1tp+2sp
            (1, 1, 2, 1, 1, 2),  # 1tp+2sp -> 1tp+2sp
            (1, 1, 2, 1, 2, 1),  # 1tp+2sp -> 1tp+1sp
            (2, 1, 1, 1, 1, 2),  # 2tp+1sp -> 1tp+2sp
            (2, 1, 2, 1, 2, 2),  # 2tp+2sp -> 1tp+2sp
            (2, 1, 2, 2, 1, 2),  # 2tp+2sp -> 2tp+2sp
        ],
    )
    def test(
        self,
        vision_model_name: str,
        language_model_name: str,
        vtp: int,
        vpp: int,
        vsp: int,
        ltp: int,
        lpp: int,
        lsp: int,
    ):
        if vsp > 2 or lsp > 2:
            pytest.skip(
                "With current data, context parallelism with more than 2 cp rank is not supported."
            )

        self.set_model(
            encoders={"vision": vision_models[vision_model_name]()},
            llm=causal_lms[language_model_name](),
        )
        self.run_multimodal_parallel(
            {"vision": vtp, "llm": ltp},
            {"vision": vpp, "llm": lpp},
            {"vision": vsp, "llm": lsp},
        )
