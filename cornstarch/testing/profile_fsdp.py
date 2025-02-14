import csv
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.device_mesh import init_device_mesh

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
    ("qwen2_72b", "vit_22b", None),  # VLM-large
    ("mixtral_8x7b", None, "qwen2_audio"),  # ALM-large
    ("llama_70b", "evaclip_18b", "whisper_1.5b"),  # VALM-large
    ("gemma2_27b", "intern_vit_6b", None),  # VLM-medium
    ("internlm2_20b", None, "whisper_307m"),  # ALM-medium
    ("qwen2_14b", "qwen2_vision_675m", "whisper_307m"),  # VALM-medium
    ("llama_8b", "pixtral_400m", None),  # VLM-small
    ("phi3_small", None, "whisper_242m"),  # ALM-small
    ("mistral_7b", "siglip_878m", "whisper_242m"),  # VALM-small
]


class FSDPTestingClass:
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
            data.update(
                self.encoder_model_classes["vision"].data(
                    self.microbatch_size, image_size=(1280, 720)
                )
            )
        if "audio" in self.encoder_model_classes:
            data.update(self.encoder_model_classes["audio"].data(self.microbatch_size))

        return data

    def build_model(self) -> nn.Module:
        with torch.device("meta"):
            encoders = {
                key: encoder_class.build_model()
                for key, encoder_class in self.encoder_model_classes.items()
            }

            model = MultimodalModel(
                encoders=encoders,
                language_model=self.llm_model_class.build_model(),
            ).to(dtype=torch.bfloat16)

        for encoder in encoders.values():
            encoder.train(module=False, projector=True)
            encoder.module.requires_grad_(False)
            encoder.projector.requires_grad_(True)
            try:
                # use_reentrant=False is neccesary for gradient checkpointing + partial freeze
                encoder.module.gradient_checkpointing_enable({"use_reentrant": False})
            except ValueError:
                # If some model doesn't support gradient checkpointing
                pass

        model.language_model.train(mode=False)
        model.language_model.requires_grad_(False)
        try:
            # use_reentrant=False is neccesary for gradient checkpointing + partial freeze
            model.language_model.gradient_checkpointing_enable({"use_reentrant": False})
        except ValueError:
            # If some model doesn't support gradient checkpointing
            pass
        token_ids = {
            "vision": 44,
            "audio": 55,
        }
        model.set_token_ids(
            {key: value for key, value in token_ids.items() if key in encoders}
        )

        device_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
        layer_lists = [
            module for module in model.modules() if isinstance(module, nn.ModuleList)
        ]
        for layer_list in layer_lists:
            for layer_id, module in enumerate(layer_list):

                reshard_after_forward = layer_id < len(layer_list) - 1

                fully_shard(
                    module,
                    reshard_after_forward=reshard_after_forward,
                    mesh=device_mesh,
                )
        for encoder in model.encoders.values():
            fully_shard(encoder, reshard_after_forward=True, mesh=device_mesh)
        fully_shard(model, reshard_after_forward=True, mesh=device_mesh)

        model.to_empty(device="cuda")

        return model

    def microbatch_forward(self, model: nn.Module, data: dict):
        with torch.autograd.profiler.emit_nvtx():
            model.set_requires_gradient_sync(True)
            with torch.cuda.nvtx.range("forward"):
                output = model(**data)
                loss = output.loss
            with torch.cuda.nvtx.range("backward"):
                loss.backward()

    def run_multimodal_model(self, model: nn.Module):
        num_iterations = 2

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        data = self.data()

        # Special data handling for pixtral/qwen2vl
        if (
            "vision" in self.encoder_model_classes
            and self.encoder_model_classes["vision"].__class__ == Qwen2Vision7bClass
        ):
            data["pixel_values"].squeeze_(0)
            data["image_grid_thw"].squeeze_(0)

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

    measure_name = f"{__name__}.FSDPTestingClass.microbatch_forward"
    with manager.measure(measure_name):
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
        elapsed_times = manager.get_elapsed_times()[measure_name]
        average_exec_time = sum(elapsed_times[1:]) / (len(elapsed_times) - 1)

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

            command = multinode_command + [
                f"--llm_model_name={llm_model_name}",
            ]

            if vision_model_name:
                command.append(f"--vision_model_name={vision_model_name}")
            if audio_model_name:
                command.append(f"--audio_model_name={audio_model_name}")

            print(f"Running: {' '.join(command)}")
            time.sleep(5)
            result = subprocess.run(command)
            # if result.returncode > 0:
            #     print(f"Error running command: {' '.join(command)}")
            #     sys.exit(1)
