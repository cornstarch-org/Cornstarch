import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.shardformer.shard.shard_config import ContextParallelDistributionMode

from .model_zoo import (
    CLIPModelBase,
    Dinov2ModelBase,
    EvaCLIPModelBase,
    Gemma2ForCausalLMBase,
    Gemma2ModelBase,
    GemmaForCausalLMBase,
    GemmaModelBase,
    InternVisonModelBase,
    LlamaForCausalLMBase,
    LlamaModelBase,
    MistralForCausalLMBase,
    MistralModelBase,
    MixtralForCausalLMBase,
    MixtralModelBase,
    Phi3ForCausalLMBase,
    Phi3ModelBase,
    PixtralVisionModelBase,
    Qwen2AudioEncoderBase,
    Qwen2ForCausalLMBase,
    Qwen2ModelBase,
    Qwen2VisionTransformerBase,
    SiglipModelBase,
    ViTModelBase,
    WhisperEncoderBase,
)
from .utils import ColossalaiHybridParallelBase

vision_models = dict(
    clip=CLIPModelBase,
    dinov2=Dinov2ModelBase,
    evaclip=EvaCLIPModelBase,
    siglip=SiglipModelBase,
    pixtral=PixtralVisionModelBase,
    qwen2_vision=Qwen2VisionTransformerBase,
    vit=ViTModelBase,
)

language_models = dict(
    gemma=GemmaModelBase,
    gemma2=Gemma2ModelBase,
    llama=LlamaModelBase,
    mistral=MistralModelBase,
    mixtral=MixtralModelBase,
    phi3=Phi3ModelBase,
    qwen2=Qwen2ModelBase,
)

causal_lms = dict(
    gemma=GemmaForCausalLMBase,
    gemma2=Gemma2ForCausalLMBase,
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
    @parametrize(
        "attention",
        ["flash_attention_2", "eager"],
        name_fn=lambda x: {
            "flash_attention_2": "fa",
            "eager": "eager",
        }[x],
    )
    @parametrize("precision", ["bf16"], name_fn=lambda p: p)
    def test(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
        attention: str,
        precision: str,
    ):
        self.set_model(vision_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, attention, precision)

    def postprocess_data_for_sharded_model(self, data, precision):
        if not isinstance(self.model, Qwen2VisionTransformerBase):
            return super().postprocess_data_for_original_model(data, precision)

        # Special data handling for Qwen2Vision
        num_batch = self.num_microbatches * self.microbatch_size
        new_data = {
            "pixel_values": torch.stack(
                data["hidden_states"].chunk(num_batch, dim=0), dim=0
            ),
            "image_grid_thw": torch.stack(
                data["grid_thw"].chunk(num_batch, dim=0), dim=0
            ),
        }
        return super().postprocess_data_for_original_model(new_data, precision)

    @parametrize("model_name", vision_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    def test_context_parallel(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
    ):
        self.set_model(vision_models[model_name]())
        self.run_hybrid_parallel(
            tp_size,
            pp_size,
            "flash_attention_2",
            "bf16",
            "ring_attn",
        )


@instantiate_parametrized_tests
class LanguageHybridParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", language_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize(
        "attention",
        ["bitfield_attention", "flash_attention_2", "eager"],
        name_fn=lambda x: {
            "bitfield_attention": "bam",
            "flash_attention_2": "fa",
            "eager": "eager",
        }[x],
    )
    @parametrize("precision", ["bf16"], name_fn=lambda p: p)
    def test_model(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
        attention: str,
        precision: str,
    ):
        self.set_model(language_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, attention, precision)

    @parametrize("model_name", causal_lms.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize(
        "attention",
        ["bitfield_attention", "flash_attention_2", "eager"],
        name_fn=lambda x: {
            "bitfield_attention": "bam",
            "flash_attention_2": "fa",
            "eager": "eager",
        }[x],
    )
    @parametrize("precision", ["bf16"], name_fn=lambda p: p)
    def test_causal(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
        attention: str,
        precision: str,
    ):
        self.set_model(causal_lms[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, attention, precision)

    @parametrize("model_name", causal_lms.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize(
        "context_parallel_dist_mode",
        [
            ContextParallelDistributionMode.UNIFORM,
            ContextParallelDistributionMode.ZIGZAG,
            ContextParallelDistributionMode.MAKESPAN_MIN,
        ],
        name_fn=lambda x: {
            ContextParallelDistributionMode.UNIFORM: "uniform",
            ContextParallelDistributionMode.ZIGZAG: "zigzag",
            ContextParallelDistributionMode.MAKESPAN_MIN: "makespan_min",
        }[x],
    )
    def test_context_parallel(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
        context_parallel_dist_mode: ContextParallelDistributionMode,
    ):
        self.set_model(causal_lms[model_name]())
        self.run_hybrid_parallel(
            tp_size,
            pp_size,
            "bitfield_attention",
            "bf16",
            "ring_attn",
            context_parallel_dist_mode,
        )


@instantiate_parametrized_tests
class AudioHybridParallel(ColossalaiHybridParallelBase):
    @parametrize("model_name", audio_models.keys(), name_fn=lambda m: m)
    @parametrize(
        "tp_size, pp_size",
        [(4, 1), (1, 1), (2, 2), (1, 4)],
        name_fn=lambda tp, pp: f"tp{tp}_pp{pp}",
    )
    @parametrize(
        "attention",
        ["flash_attention_2", "eager"],
        name_fn=lambda x: {
            "flash_attention_2": "fa",
            "eager": "eager",
        }[x],
    )
    @parametrize("precision", ["bf16"], name_fn=lambda p: p)
    def test(
        self,
        model_name: str,
        tp_size: int,
        pp_size: int,
        attention: str,
        precision: str,
    ):
        self.set_model(audio_models[model_name]())
        self.run_hybrid_parallel(tp_size, pp_size, attention, precision)
