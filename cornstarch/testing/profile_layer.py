import csv
import os
from typing import Any

import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin.hybrid_parallel_plugin import (
    HybridParallelModule,
    HybridParallelPlugin,
)
from colossalai.shardformer.policies.auto_policy import _fullname

from cornstarch.shardformer.policies.auto_policy import get_autopolicy
from cornstarch.testing.model_zoo import (
    CLIPVisionClass,
    Dinov2BaseClass,
    Dinov2GiantClass,
    Dinov2LargeClass,
    Dinov2SmallClass,
    EvaCLIPVision8bClass,
    EvaCLIPVision18bClass,
    Gemma2bClass,
    Gemma7bClass,
    InternLM27bClass,
    InternVision6bClass,
    InternVision300mClass,
    LanguageModelClassBase,
    Llama1bClass,
    Llama3bClass,
    Llama8bClass,
    Llama70bClass,
    Mistral7bClass,
    ModelClassBase,
    Phi3MiniClass,
    Phi3SmallClass,
    PixtralVisionClass,
    Qwen2AudioEncoderClass,
    Qwen2Vision7bClass,
    Qwen23bClass,
    Qwen27bClass,
    Qwen205bClass,
    Qwen214bClass,
    Qwen215bClass,
    Qwen272bClass,
    SiglipVisionClass,
    Vicuna7bClass,
    WhisperBaseClass,
    WhisperLargeClass,
    WhisperSmallClass,
)
from cornstarch.testing.timer import TimerContextManager

model_to_class = {
    # "gemma_2b": Gemma2bClass,
    # "gemma_7b": Gemma7bClass,
    # "llama_1b": Llama1bClass,
    # "llama_3b": Llama3bClass,
    "llama_8b": Llama8bClass,
    # "llama_70b": Llama70bClass,
    # "internlm2": InternLM27bClass,
    # "mistral_7b": Mistral7bClass,
    # "phi3_mini": Phi3MiniClass,
    # "phi3_small": Phi3SmallClass,
    # "qwen2_0.5b": Qwen205bClass,
    # "qwen2_1.5b": Qwen215bClass,
    # "qwen2_3b": Qwen23bClass,
    # "qwen2_7b": Qwen27bClass,
    # "qwen2_14b": Qwen214bClass,
    # "qwen2_72b": Qwen272bClass,
    # "vicuna": Vicuna7bClass,
    # "clip": CLIPVisionClass,
    # "evaclip_8b": EvaCLIPVision8bClass,
    # "evaclip_18b": EvaCLIPVision18bClass,
    # "dinov2_22m": Dinov2SmallClass,
    # "dinov2_86m": Dinov2BaseClass,
    # "dinov2_300m": Dinov2LargeClass,
    # "dinov2_1.1b": Dinov2GiantClass,
    # "intern_vit_300m": InternVision300mClass,
    # "intern_vit_6b": InternVision6bClass,
    # "pixtral_400m": PixtralVisionClass,
    # "qwen2_vision_675m": Qwen2Vision7bClass,
    # "siglip_878m": SiglipVisionClass,
    # "whisper_1.5b": WhisperLargeClass,
    # "whisper_242m": WhisperSmallClass,
    # "whisper_72m": WhisperBaseClass,
    # "qwen2_audio": Qwen2AudioEncoderClass,
}

class_to_forward_str = {
    Gemma2bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Gemma7bClass: "cornstarch.shardformer.modeling.gemma.GemmaModelForwards.gemma_model_forward",
    Llama1bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama3bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama8bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    Llama70bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    InternLM27bClass: "cornstarch.shardformer.modeling.internlm2.InternLM2ModelForwards.internlm2_model_forward",
    Mistral7bClass: "cornstarch.shardformer.modeling.mistral.MistralModelForwards.mistral_model_forward",
    Phi3MiniClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Phi3SmallClass: "cornstarch.shardformer.modeling.phi3.Phi3ModelForwards.phi3_model_forward",
    Qwen205bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen215bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen23bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen27bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen214bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Qwen272bClass: "cornstarch.shardformer.modeling.qwen2.Qwen2ModelForwards.qwen2_model_forward",
    Vicuna7bClass: "cornstarch.shardformer.modeling.llama.LlamaModelForwards.llama_model_forward",
    CLIPVisionClass: "cornstarch.shardformer.modeling.clip.CLIPVisionModelForwards.clip_vision_transformer_forward",
    EvaCLIPVision8bClass: "cornstarch.shardformer.modeling.evaclip.EvaCLIPModelForwards.eva_clip_vision_transformer_forward",
    EvaCLIPVision18bClass: "cornstarch.shardformer.modeling.evaclip.EvaCLIPModelForwards.eva_clip_vision_transformer_forward",
    Dinov2SmallClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2BaseClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2LargeClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    Dinov2GiantClass: "cornstarch.shardformer.modeling.dinov2.Dinov2ModelForwards.dinov2_encoder_forward",
    InternVision300mClass: "cornstarch.shardformer.modeling.intern_vit.InternVisionModelForwards.intern_vit_model_forward",
    InternVision6bClass: "cornstarch.shardformer.modeling.intern_vit.InternVisionModelForwards.intern_vit_model_forward",
    PixtralVisionClass: "cornstarch.shardformer.modeling.pixtral.PixtralVisionModelForwards.pixtral_vision_model_forward",
    Qwen2Vision7bClass: "cornstarch.shardformer.modeling.qwen2_vision.Qwen2VisionModelForwards.qwen2_vision_transformer_forward",
    SiglipVisionClass: "cornstarch.shardformer.modeling.siglip.SiglipVisionModelForwards.siglip_vision_transformer_forward",
    WhisperSmallClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    WhisperBaseClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    WhisperLargeClass: "cornstarch.shardformer.modeling.whisper.WhisperModelForwards.whisper_encoder_forward",
    Qwen2AudioEncoderClass: "cornstarch.shardformer.modeling.qwen2_audio.Qwen2AudioModelForwards.qwen2_audio_encoder_forward",
}

file_path = "profile_layer_result.csv"


class CornstarchTestingClass:
    num_microbatches: int = 1
    microbatch_size: int = 1

    def __init__(
        self,
        model_class: ModelClassBase,
    ):
        self.model_class = model_class

    def data(self) -> dict[str, torch.Tensor]:
        data = {}
        batch_size = self.num_microbatches * self.microbatch_size
        if isinstance(self.model_class, LanguageModelClassBase):
            data.update(self.model_class.data(batch_size, seq_len=8192))
        else:
            data.update(self.model_class.data(batch_size))

        return data

    def build_model(
        self,
        tp_size: int,
        test_config: dict[str, Any],
        llm_sp_size: int = None,
    ) -> tuple[HybridParallelModule, Booster]:
        test_config.update(
            {
                "num_microbatches": self.num_microbatches,
                "microbatch_size": self.microbatch_size,
                "enable_flash_attention": False,
                "tp_size": tp_size,
                "pp_size": 1,
                "zero_stage": 0,
            }
        )
        if llm_sp_size is not None and llm_sp_size > 1:
            test_config.update(
                {
                    "enable_sequence_parallelism": True,
                    "sequence_parallelism_mode": "ring_attn",
                    "sp_size": llm_sp_size,
                }
            )

        self.model_class.config.num_hidden_layers = 4
        model = self.model_class.build_model().to(dtype=torch.bfloat16)

        policy = get_autopolicy(_fullname(model))

        plugin = HybridParallelPlugin(custom_policy=policy, **test_config)
        plugin.precision = None
        setattr(plugin.shard_config, "ring_attention_distribution_mode", "zigzag")

        booster = Booster(plugin=plugin)
        model, *_ = booster.boost(model)

        return model, booster

    def run_model(self, model: HybridParallelModule, booster: Booster):
        num_iterations = 5

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        data = self.data()

        with torch.no_grad():
            for _ in range(num_iterations):
                model(**data)


def run_profile(model_name: str, tp_size: int, sp_size: int):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="nccl",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    cornstarch_class = CornstarchTestingClass(model_to_class[model_name]())

    manager = TimerContextManager()

    with manager.measure(class_to_forward_str[cornstarch_class.model_class.__class__]):

        model, booster = cornstarch_class.build_model(
            tp_size=tp_size, llm_sp_size=sp_size, test_config=dict()
        )

        dist.barrier()

        cornstarch_class.run_model(model, booster)

        torch.cuda.synchronize()

    if dist.get_rank() == 0:
        elapsed_times = manager.get_elapsed_times()
        average_forward_time = sum(time for _, time in elapsed_times[1:]) / (
            len(elapsed_times) - 1
        )
        peak_memory_bytes = torch.cuda.memory_stats()["active_bytes.all.peak"]
        peak_memory_gib = peak_memory_bytes / (1024**3)

        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "tp_size",
                    "sp_size",
                    "forward_time (ms)",
                    "peak_memory (GiB)",
                ],
            )
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(
                {
                    "model": model_name,
                    "tp_size": tp_size,
                    "sp_size": sp_size,
                    "forward_time (ms)": average_forward_time,
                    "peak_memory (GiB)": peak_memory_gib,
                }
            )

    dist.barrier()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # If LOCAL_RANK is set, we are in a child process
        import argparse

        parser = argparse.ArgumentParser(
            description="Run profile with distributed processes"
        )
        parser.add_argument("--model_name", type=str, help="Model name")
        parser.add_argument("--tp_size", type=int, help="Tensor parallel size")
        parser.add_argument("--sp_size", type=int, help="LLM SP size")
        args = parser.parse_args()

        torch._dynamo.config.optimize_ddp = False
        run_profile(
            model_name=args.model_name, tp_size=args.tp_size, sp_size=args.sp_size
        )
        torch.cuda.synchronize()
    else:
        import subprocess
        import sys

        # If LOCAL_RANK is not set, we are in the main process and need to launch child processes
        num_gpus = 4  # Set this to the number of GPUs you want to use

        for model_name in model_to_class.keys():
            for tp_size, sp_size in [(1, 4)]:
                command = [
                    "torchrun",
                    "--nproc_per_node",
                    str(num_gpus),
                    sys.argv[0],  # The current script file
                    "--model_name",
                    model_name,
                    "--tp_size",
                    str(tp_size),
                    "--sp_size",
                    str(sp_size),
                ]

                print(f"Running: {' '.join(command)}")
                subprocess.run(command)
                # result = subprocess.run(
                #     command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                # )
