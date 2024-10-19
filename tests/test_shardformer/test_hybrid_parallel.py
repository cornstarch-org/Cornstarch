from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from .model_zoo import (
    CLIPModelBase,
    Dinov2ModelBase,
    EvaCLIPModelBase,
    Gemma2ForCausalLMBase,
    Gemma2ModelBase,
    GemmaForCausalLMBase,
    GemmaModelBase,
    InternLM2ForCausalLMBase,
    InternLM2ModelBase,
    InternVisonModelBase,
    LlamaForCausalLMBase,
    LlamaModelBase,
    MistralForCausalLMBase,
    MistralModelBase,
    MixtralForCausalLMBase,
    MixtralModelBase,
    Phi3ForCausalLMBase,
    Phi3ModelBase,
    Qwen2AudioEncoderBase,
    Qwen2ForCausalLMBase,
    Qwen2ModelBase,
    SiglipModelBase,
    WhisperEncoderBase,
)
from .utils import ColossalaiHybridParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    dinov2=Dinov2ModelBase,
    evaclip=EvaCLIPModelBase,
    siglip=SiglipModelBase,
    intern_vit=InternVisonModelBase,
)

language_models = dict(
    gemma=GemmaModelBase,
    gemma2=Gemma2ModelBase,
    internlm2=InternLM2ModelBase,
    llama=LlamaModelBase,
    mistral=MistralModelBase,
    mixtral=MixtralModelBase,
    phi3=Phi3ModelBase,
    qwen2=Qwen2ModelBase,
)

causal_lms = dict(
    gemma=GemmaForCausalLMBase,
    gemma2=Gemma2ForCausalLMBase,
    internlm2=InternLM2ForCausalLMBase,
    llama=LlamaForCausalLMBase,
    mistral=MistralForCausalLMBase,
    mixtral=MixtralForCausalLMBase,
    phi3=Phi3ForCausalLMBase,
    qwen2=Qwen2ForCausalLMBase,
)

audio_models = dict(
    qwen2_audio=Qwen2AudioEncoderBase,
    whisper=WhisperEncoderBase,
)


@instantiate_parametrized_tests
class VisionHybridParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", vision_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self, model_name: str, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(vision_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)


@instantiate_parametrized_tests
class LanguageHybridParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", language_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test_model(
        self, model_name: str, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(language_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)

    @parametrize("model_name", causal_lms.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test_causal(
        self, model_name: str, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(causal_lms[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)


@instantiate_parametrized_tests
class LanguageContextParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", language_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    def test_model(self, model_name: str, tp_size: int, pp_size: int):
        self.set_model(language_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, "all_to_all", True, "bf16")

    @parametrize("model_name", causal_lms.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("sp_mode", ["all_to_all", "ring_attn"], name_fn=lambda x: x)
    def test_causal(self, model_name: str, tp_size: int, pp_size: int, sp_mode: str):
        self.set_model(causal_lms[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, sp_mode, True, "bf16")


@instantiate_parametrized_tests
class AudioHybridParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", audio_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize("fa", [True, False])
    @parametrize("precision", ["bf16", "fp16"], name_fn=lambda p: p)
    def test(
        self, model_name: str, tp_size: int, pp_size: int, fa: bool, precision: str
    ):
        self.set_model(audio_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, None, fa, precision)
