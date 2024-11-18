import csv
import os
import socket

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard

from cornstarch.models.multimodal_language_model import (
    MultimodalModel,
)
from cornstarch.testing.model_zoo import (
    AudioModelClassBase,
    ImageModelClassBase,
    LanguageModelClassBase,
    Qwen2Vision7bClass,
    model_to_class,
)
from cornstarch.testing.nodes_info import master_node_rdzv_backend, node_hostnames
from cornstarch.testing.timer import TimerContextManager

file_path = "profile_fsdp_result.csv"

# model_names_to_test = [("llama_8b", "clip", "qwen2_audio")]
model_names_to_test = [
    ("llama_70b", "clip", None),
    ("gemma2_27b", "siglip_878m", None),
    ("internlm2_20b", "intern_vit_6b", None),
    ("mistral_7b", "pixtral_400m", None),
    ("phi3_small", "evaclip_8b", None),
    ("vicuna", "dinov2_1.1b", None),
    ("qwen2_1.5b", "qwen2_vision_675m", None),
    ("qwen2_72b", None, "qwen2_audio"),
    ("llama_8b", None, "whisper_1.5b"),
    ("mixtral_8x7b", "qwen2_vision_675m", "qwen2_audio"),
    ("qwen2_14b", "clip", "whisper_1.5b"),
]


class FSDPTestingClass:
    num_microbatches: int = 20
    microbatch_size: int = 1

    def __init__(
        self,
        llm_model_class: LanguageModelClassBase,
        encoder_model_classes: dict[str, ImageModelClassBase | AudioModelClassBase],
    ):
        self.llm_model_class = llm_model_class
        self.encoder_model_classes = encoder_model_classes

    def data(self) -> dict[str, torch.Tensor]:
        data = {}
        data.update(self.llm_model_class.data(self.microbatch_size, seq_len=4096))

        if "vision" in self.encoder_model_classes:
            data.update(self.encoder_model_classes["vision"].data(self.microbatch_size))
        if "audio" in self.encoder_model_classes:
            data.update(self.encoder_model_classes["audio"].data(self.microbatch_size))

        return data

    def build_model(self) -> nn.Module:
        with torch.device("meta"):
            encoders = {
                key: encoder_class.build_model()
                for key, encoder_class in self.encoder_model_classes.items()
            }

            if (
                "vision" in encoders
                and self.encoder_model_classes["vision"].__class__ == Qwen2Vision7bClass
            ):
                module = encoders["vision"]

            model = MultimodalModel(
                encoders=encoders,
                language_model=self.llm_model_class.build_model(),
            ).to(dtype=torch.bfloat16)

        for encoder in encoders.values():
            encoder.train(module=False, projector=True)
            encoder.module.requires_grad_(False)
            if encoder.projector is not None:
                encoder.projector.requires_grad_(True)

        model.language_model.train(mode=False)
        model.language_model.requires_grad_(False)
        token_ids = {
            "vision": 44,
            "audio": 55,
        }
        model.set_token_ids(
            {key: value for key, value in token_ids.items() if key in encoders}
        )

        layer_lists = [
            module for module in model.modules() if isinstance(module, nn.ModuleList)
        ]

        for layer_list in layer_lists:
            for layer_id, module in enumerate(layer_list):

                reshard_after_forward = layer_id < len(layer_list) - 1

                fully_shard(module, reshard_after_forward=reshard_after_forward)
        for encoder in model.encoders.values():
            fully_shard(encoder, reshard_after_forward=True)
        fully_shard(model, reshard_after_forward=True)

        model.to_empty(device="cuda")

        return model

    def microbatch_forward(self, model: nn.Module, data: dict):
        assert self.num_microbatches % dist.get_world_size() == 0
        num_microbatches_per_gpu = self.num_microbatches // dist.get_world_size()
        with torch.autograd.profiler.emit_nvtx():
            for i in range(num_microbatches_per_gpu):
                model.set_requires_gradient_sync(i == num_microbatches_per_gpu - 1)
                with torch.cuda.nvtx.range("forward"):
                    output = model(**data)
                    loss = output.loss
                with torch.cuda.nvtx.range("backward"):
                    loss.backward()

    def run_multimodal_model(self, model: nn.Module):
        num_iterations = 5

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        data = self.data()

        for i in range(num_iterations):
            print(f"Iteration {i}")
            self.microbatch_forward(model, data)


def run_profile(
    llm_model_name: str,
    vision_model_name: str | None = None,
    audio_model_name: str | None = None,
):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        backend="nccl",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    encoder_classes = {}
    if vision_model_name is not None:
        encoder_classes["vision"] = model_to_class[vision_model_name]()
    if audio_model_name is not None:
        encoder_classes["audio"] = model_to_class[audio_model_name]()

    fsdp_class = FSDPTestingClass(
        llm_model_class=model_to_class[llm_model_name](),
        encoder_model_classes=encoder_classes,
    )

    model = fsdp_class.build_model()

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.barrier(device_ids=[local_rank])

    manager = TimerContextManager()

    with manager.measure(f"{__name__}.FSDPTestingClass.microbatch_forward"):
        fsdp_class.run_multimodal_model(model)

        torch.cuda.synchronize()

    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated(f"cuda:{local_rank}") / 1024**3,
        dtype=torch.float32,
        device="cuda",
    )
    gathered_peak_memory = torch.zeros(
        dist.get_world_size(), dtype=torch.float32, device="cuda"
    )
    dist.all_gather_into_tensor(gathered_peak_memory, peak_memory)
    max_peak_memory = gathered_peak_memory.max().item()
    min_peak_memory = gathered_peak_memory.min().item()
    avg_peak_memory = gathered_peak_memory.mean().item()

    if local_rank == 0:
        elapsed_times = manager.get_elapsed_times()
        average_exec_time = sum(time for _, time in elapsed_times[1:]) / (
            len(elapsed_times) - 1
        )

        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "llm_model",
                    "vision_model",
                    "audio_model",
                    "exec_time (ms)",
                    "peak_memory (GB)",
                    "min_peak_memory (GB)",
                    "max_peak_memory (GB)",
                ],
            )
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(
                {
                    "llm_model": llm_model_name,
                    "vision_model": vision_model_name,
                    "audio_model": audio_model_name,
                    "exec_time (ms)": average_exec_time,
                    "peak_memory (GB)": avg_peak_memory,
                    "min_peak_memory (GB)": min_peak_memory,
                    "max_peak_memory (GB)": max_peak_memory,
                }
            )

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # If LOCAL_RANK is set, we are in a child process
        import argparse

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        parser = argparse.ArgumentParser(
            description="Run profile with distributed processes"
        )
        parser.add_argument("--llm_model_name", type=str, help="LLM model name")
        parser.add_argument(
            "--vision_model_name", type=str, default=None, help="Vision model name"
        )
        parser.add_argument(
            "--audio_model_name", type=str, default=None, help="Audio model name"
        )
        args = parser.parse_args()

        kargs = {"llm_model_name": args.llm_model_name}

        if args.vision_model_name is not None:
            kargs["vision_model_name"] = args.vision_model_name
        if args.audio_model_name is not None:
            kargs["audio_model_name"] = args.audio_model_name

        torch._dynamo.config.optimize_ddp = False
        run_profile(**kargs)
        torch.cuda.synchronize()
    else:
        # If LOCAL_RANK is not set, we are in the main process and need to launch child processes
        import subprocess
        import sys

        for llm_model_name, vision_model_name, audio_model_name in model_names_to_test:
            if vision_model_name is None and audio_model_name is None:
                continue

            standalone_command = [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                sys.argv[0],  # The current script file
            ]

            multinode_command = [
                "torchrun",
                f"--nnodes={len(node_hostnames)}",
                "--nproc_per_node=4",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={master_node_rdzv_backend}",
                f"--node-rank={node_hostnames.index(socket.gethostname())}",
                sys.argv[0],  # The current script file
            ]

            command = standalone_command + [
                f"--llm_model_name={llm_model_name}",
            ]

            if vision_model_name:
                command.extend(["--vision_model_name", vision_model_name])
            if audio_model_name:
                command.extend(["--audio_model_name", audio_model_name])

            print(f"Running: {' '.join(command)}")
            subprocess.run(command)
