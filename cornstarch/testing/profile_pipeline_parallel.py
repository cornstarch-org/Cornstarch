import csv
import math
import os
import re
import time
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from torch.optim import Optimizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.pipeline_template import PipelineTemplate

# from cornstarch.plugin.multimodal_parallel_plugin import (
#     ModalParallelPlugin,
#     MultimodalParallelModule,
#     MultimodalParallelPlugin,
# )
from cornstarch.plugin.encoders_colocated_plugin.encoders_colocated_plugin import (
    EncodersColocatedMultimodalParallelPlugin,
    ModalParallelPlugin,
    MultimodalParallelModule,
)
from cornstarch.testing.dcgm import DcgmContextManager
from cornstarch.testing.model_zoo import (
    AudioModelClassBase,
    ImageModelClassBase,
    LanguageModelClassBase,
    model_to_class,
)
from cornstarch.testing.nodes_info import master_node_rdzv_backend, node_hostnames
from cornstarch.testing.timer import TimerContextManager

file_path = "profile_pipeline_result.csv"

# model_names_to_test = [("llama_70b", "clip", "qwen2_audio")]
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

colocate_parallel_configuration = [
    # Megatron-like configuration without considering freeze
    # (llm_pp_size, vision_pp_size, audio_pp_size)
    (4, 2, 2),  # VLM-large
    (5, 1, 1),
    (4, 2, 2),
    (4, 2, 2),  # VLM-medium
    (5, 1, 1),
    (3, 3, 3),
    (5, 1, 1),  # VLM-small
    (5, 1, 1),
    (5, 1, 1),
]

cornstarch_parallel_configuration = [
    # Megatron-like configuration without considering freeze
    # (llm_pp_size, vision_pp_size, audio_pp_size)
    (5, 1, 1),  # VLM-large
    (5, 1, 1),
    (5, 1, 1),
    (5, 1, 1),  # VLM-medium
    (5, 1, 1),
    (5, 1, 1),
    (5, 1, 1),  # VLM-small
    (5, 1, 1),
    (5, 1, 1),
]


class CornstarchTestingClass:
    num_microbatches: int = 24
    microbatch_size: int = 1

    def __init__(
        self,
        llm_model_class: LanguageModelClassBase,
        encoder_model_classes: dict[str, ImageModelClassBase | AudioModelClassBase],
    ):
        self.llm_model_class = llm_model_class
        self.encoder_model_classes = encoder_model_classes
        self.iteration = 0

        data = self.data()
        self.data_iter = iter([data] * 10)

    def data(self) -> dict[str, torch.Tensor]:
        seq_length = 1024

        data = {}
        batch_size = self.num_microbatches * self.microbatch_size
        data.update(self.llm_model_class.data(batch_size, seq_len=seq_length))

        total_length = seq_length
        if "vision" in self.encoder_model_classes:
            image_size = (1280, 720)
            vision_data = self.encoder_model_classes["vision"].data(
                batch_size, image_size=image_size
            )
            num_tokens_to_inject = self.encoder_model_classes["vision"].get_num_tokens(
                batch_size, image_size
            )
            start_point = seq_length // 4

            data["input_ids"] = torch.cat(
                [
                    data["input_ids"][:, :start_point],
                    torch.full(
                        (batch_size, num_tokens_to_inject), 44, dtype=torch.long
                    ),
                    data["input_ids"][:, start_point:],
                ],
                dim=1,
            )
            data["labels"] = torch.cat(
                [
                    data["labels"][:, :start_point],
                    torch.full(
                        (batch_size, num_tokens_to_inject), -100, dtype=torch.long
                    ),
                    data["labels"][:, start_point:],
                ],
                dim=1,
            )
            data.update(vision_data)
            total_length += num_tokens_to_inject

        if "audio" in self.encoder_model_classes:
            audio_data = self.encoder_model_classes["audio"].data(batch_size)
            num_tokens_to_inject = self.encoder_model_classes["audio"].get_num_tokens()
            start_point = total_length // 4 * 3

            data["input_ids"] = torch.cat(
                [
                    data["input_ids"][:, :start_point],
                    torch.full(
                        (batch_size, num_tokens_to_inject), 55, dtype=torch.long
                    ),
                    data["input_ids"][:, start_point:],
                ],
                dim=1,
            )
            data["labels"] = torch.cat(
                [
                    data["labels"][:, :start_point],
                    torch.full(
                        (batch_size, num_tokens_to_inject), -100, dtype=torch.long
                    ),
                    data["labels"][:, start_point:],
                ],
                dim=1,
            )

            data.update(audio_data)
            total_length += num_tokens_to_inject

        remainder = total_length % 4
        if remainder != 0:
            data["input_ids"] = data["input_ids"][:, :-remainder]
            data["labels"] = data["labels"][:, :-remainder]

        return data

    @staticmethod
    def get_megatron_style_pipeline_template(
        model: nn.Module, num_stages: int
    ) -> PipelineTemplate:
        modules = PipelineTemplate.get_modules(model)
        num_layers = sum(bool(re.search(r"\.\d", s)) for s in modules)

        # Get the number of layers per stage
        base_size = num_layers // num_stages
        remainder = num_layers % num_stages
        num_layers_per_stage = [
            base_size + 1 if i < remainder else base_size for i in range(num_stages)
        ]
        assert sum(num_layers_per_stage) == num_layers

        first_layer_index = next(
            i for i, layer in enumerate(modules) if re.search(r"\.0", layer)
        )
        last_layer_index = next(
            i
            for i, layer in enumerate(modules)
            if re.search(rf"\.{num_layers - 1}", layer)
        )

        modules_per_stages = [[] for _ in range(num_stages)]
        modules_per_stages[0].extend(modules[:first_layer_index])
        layer_idx = 0
        for stage_idx, num_layers in enumerate(num_layers_per_stage):
            idx = first_layer_index + layer_idx
            modules_per_stages[stage_idx].extend(modules[idx : idx + num_layers])
            layer_idx += num_layers
        modules_per_stages[-1].extend(modules[last_layer_index + 1 :])

        return PipelineTemplate(
            (
                model.config[0].model_type
                if isinstance(model, ModalEncoderModule)
                else model.config.model_type
            ),
            modules_per_stages,
        )

    def build_model(
        self,
        tp_size: int,
        llm_pp_size: int,
        encoders_pp_size: dict[str, int],
        test_config: dict[str, Any],
    ) -> tuple[MultimodalParallelModule, Optimizer, Callable, Booster]:
        test_config.update(
            {
                "num_microbatches": self.num_microbatches,
                "microbatch_size": self.microbatch_size,
                "enable_flash_attention": False,
            }
        )

        with torch.device("meta"):
            encoders = {
                key: encoder_class.build_model()
                for key, encoder_class in self.encoder_model_classes.items()
            }

            model = MultimodalModel(
                encoders=encoders,
                language_model=self.llm_model_class.build_model(),
            )

        token_ids = {
            "vision": 44,
            "audio": 55,
        }
        model.set_token_ids(
            {key: value for key, value in token_ids.items() if key in encoders}
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        llm_plugin = ModalParallelPlugin(
            tp_size=tp_size,
            sequence_parallelism_mode=None,
            pipeline_template=self.get_megatron_style_pipeline_template(
                model.language_model, llm_pp_size
            ),
        )

        encoders_plugins = {
            key: ModalParallelPlugin(
                tp_size=tp_size,
                pipeline_template=self.get_megatron_style_pipeline_template(
                    model.get_submodule(f"{key}_encoder"), encoders_pp_size[key]
                ),
            )
            for key in encoders
        }

        plugin = EncodersColocatedMultimodalParallelPlugin(
            encoder_plugins=encoders_plugins,
            language_model_plugin=llm_plugin,
            **test_config,
        )
        plugin.precision = None
        booster = Booster(plugin=plugin)

        def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
            return x.loss

        model, optimizer, criterion, *_ = booster.boost(model, optimizer, loss_fn)
        model.to(dtype=torch.bfloat16)

        for encoder in model.module.encoders.values():
            encoder.train(module=False, projector=True)
            for param in encoder.module.parameters():
                param.requires_grad_(False)
            for param in encoder.projector.parameters():
                param.requires_grad_(True)
            try:
                encoder.module.gradient_checkpointing_enable({"use_reentrant": False})
            except ValueError:
                pass

        model.module.language_model.train(mode=False)
        for param in model.module.language_model.parameters():
            param.requires_grad_(False)
        for buf in model.module.language_model.buffers():
            buf.requires_grad_(False)
        model.module.language_model.gradient_checkpointing_enable(
            {"use_reentrant": False}
        )

        return model, optimizer, criterion, booster

    def run_multimodal_model(
        self,
        model: MultimodalParallelModule,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        booster: Booster,
    ):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.iteration += 1

        with (
            torch.autograd.profiler.emit_nvtx(),
            torch.cuda.nvtx.range(f"iter{self.iteration}"),
        ):
            booster.execute_pipeline(
                self.data_iter,
                model,
                lambda outputs, inputs: criterion(outputs),
                optimizer,
                return_loss=True,
                return_outputs=False,
            )


def run_profile(
    llm_model_name: str,
    llm_pp_size: int,
    tp_size: int,
    vision_model_name: str | None = None,
    audio_model_name: str | None = None,
    vision_pp_size: int | None = None,
    audio_pp_size: int | None = None,
):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_device("cuda")
    dist.init_process_group(
        backend="nccl",
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )

    encoder_classes = {}
    encoder_pp_sizes = {}
    if vision_model_name is not None:
        encoder_classes["vision"] = model_to_class[vision_model_name]()
        encoder_pp_sizes["vision"] = vision_pp_size
    if audio_model_name is not None:
        encoder_classes["audio"] = model_to_class[audio_model_name]()
        encoder_pp_sizes["audio"] = audio_pp_size

    cornstarch_class = CornstarchTestingClass(
        llm_model_class=model_to_class[llm_model_name](),
        encoder_model_classes=encoder_classes,
    )

    model, optimizer, criterion, booster = cornstarch_class.build_model(
        tp_size=tp_size,
        llm_pp_size=llm_pp_size,
        encoders_pp_size=encoder_pp_sizes,
        test_config=dict(),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.barrier(device_ids=[local_rank])

    timer_manager = TimerContextManager()
    dcgm_manager = DcgmContextManager()

    fb_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.run_forward_backward"
    fwd_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.forward_step"
    bwd_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.backward_step"

    # warmup
    print("Warming up")
    cornstarch_class.run_multimodal_model(model, optimizer, criterion, booster)

    dist.barrier()
    torch.cuda.synchronize()

    num_iterations = 2
    with (
        timer_manager.measure([fb_measure_name, fwd_measure_name, bwd_measure_name]),
        dcgm_manager.profile(),
    ):
        for _ in range(num_iterations):
            print(f"Running iteration {cornstarch_class.iteration}/{num_iterations}")
            cornstarch_class.run_multimodal_model(model, optimizer, criterion, booster)

        dist.barrier()
        torch.cuda.synchronize()

    elapsed_times = timer_manager.get_elapsed_times()
    sm_occupancy_trace = dcgm_manager.get_sm_occupancy_trace()
    tensor_util_trace = dcgm_manager.get_tensor_core_util_trace()

    tp_rank = dist.get_rank(booster.plugin.tp_group)
    if tp_rank == 0:
        try:
            pp_group = booster.plugin.pp_groups[0]
        except AttributeError:
            pp_group = booster.plugin.pp_group
        fwd_times = elapsed_times[fwd_measure_name]
        average_fwd_time = torch.tensor(
            sum(fwd_times) / len(fwd_times),
            device="cuda",
        )
        average_fwd_times = torch.empty(dist.get_world_size(pp_group), device="cuda")
        dist.all_gather_into_tensor(average_fwd_times, average_fwd_time, group=pp_group)

        bwd_times = elapsed_times[bwd_measure_name]
        average_bwd_time = torch.tensor(sum(bwd_times) / len(bwd_times), device="cuda")
        average_bwd_times = torch.empty(dist.get_world_size(pp_group), device="cuda")
        dist.all_gather_into_tensor(average_bwd_times, average_bwd_time, group=pp_group)

        fb_times = elapsed_times[fb_measure_name]
        average_exec_time = sum(fb_times) / len(fb_times)

        pp_rank = dist.get_rank(pp_group)
        with open(f"pp{pp_rank}_{file_path}", "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "llm_pp",
                    "vision_pp",
                    "audio_pp",
                    "llm_model",
                    "vision_model",
                    "audio_model",
                    "exec_time (ms)",
                    "fwd_time_per_pp (ms)",
                    "bwd_time_per_pp (ms)",
                ],
            )
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(
                {
                    "llm_pp": llm_pp_size,
                    "vision_pp": vision_pp_size,
                    "audio_pp": audio_pp_size,
                    "llm_model": llm_model_name,
                    "vision_model": vision_model_name,
                    "audio_model": audio_model_name,
                    "exec_time (ms)": average_exec_time,
                    "fwd_time_per_pp (ms)": str(average_fwd_times.tolist()),
                    "bwd_time_per_pp (ms)": str(average_bwd_times.tolist()),
                }
            )

        with open(
            f"pp{pp_rank}_sm_{llm_model_name}+{vision_model_name}+{audio_model_name}.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "sm_occupancy"])
            writer.writerows(sm_occupancy_trace)

        with open(
            f"pp{pp_rank}_tensor_{llm_model_name}+{vision_model_name}+{audio_model_name}.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "tensor_active"])
            writer.writerows(tensor_util_trace)

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    dcgm_manager.shutdown()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        # If LOCAL_RANK is set, we are in a child process
        import argparse

        parser = argparse.ArgumentParser(
            description="Run profile with distributed processes"
        )
        parser.add_argument("--llm_model_name", type=str, help="LLM model name")
        parser.add_argument(
            "--llm_pp_size", type=int, help="LLM pipeline parallel size"
        )
        parser.add_argument("--tp_size", type=int, help="Tensor parallel size")
        parser.add_argument(
            "--vision_model_name", type=str, default=None, help="Vision model name"
        )
        parser.add_argument(
            "--audio_model_name", type=str, default=None, help="Audio model name"
        )
        parser.add_argument(
            "--vision_pp_size",
            type=int,
            default=None,
            help="Encoder pipeline parallel size",
        )
        parser.add_argument(
            "--audio_pp_size",
            type=int,
            default=None,
            help="Encoder pipeline parallel size",
        )
        args = parser.parse_args()

        kargs = {
            "llm_model_name": args.llm_model_name,
            "llm_pp_size": args.llm_pp_size,
            "tp_size": args.tp_size,
        }

        if args.vision_model_name is not None:
            kargs.update(
                {
                    "vision_model_name": args.vision_model_name,
                    "vision_pp_size": args.vision_pp_size,
                }
            )
        if args.audio_model_name is not None:
            kargs.update(
                {
                    "audio_model_name": args.audio_model_name,
                    "audio_pp_size": args.audio_pp_size,
                }
            )

        torch._dynamo.config.optimize_ddp = False
        run_profile(**kargs)
        torch.cuda.synchronize()
    else:
        # If LOCAL_RANK is not set, we are in the main process and need to launch child processes
        import socket
        import subprocess
        import sys

        my_node_index = node_hostnames.index(socket.gethostname())
        store = dist.TCPStore(
            host_name=node_hostnames[0],
            port=14400,
            world_size=len(node_hostnames),
            is_master=my_node_index == 0,
            wait_for_workers=True,
        )

        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=my_node_index,
            world_size=len(node_hostnames),
        )

        for model_index, (
            llm_model_name,
            vision_model_name,
            audio_model_name,
        ) in enumerate(model_names_to_test):
            if vision_model_name is None and audio_model_name is None:
                continue

            llm_pp_size, vision_pp_size, audio_pp_size = (
                colocate_parallel_configuration[model_index]
            )
            num_ranks = (llm_pp_size + vision_pp_size) * 2

            num_nodes_to_join = math.ceil(num_ranks / 4)

            if my_node_index < len(node_hostnames) - num_nodes_to_join:
                print(
                    "This node does not join the current model run. Waiting for the job to finish."
                )
            else:
                node_rank = my_node_index - (len(node_hostnames) - num_nodes_to_join)
                num_processes = 4
                if node_rank == 0 and num_ranks % 4 != 0:
                    num_processes = num_ranks % 4

                multinode_command = [
                    "torchrun",
                    f"--nnodes={num_nodes_to_join}",
                    f"--nproc_per_node={num_processes}",
                    "--rdzv_backend=c10d",
                    f"--rdzv_endpoint={master_node_rdzv_backend}",
                    f"--node-rank={node_rank}",
                    sys.argv[0],  # The current script file
                    "--tp_size=2",
                    f"--llm_pp_size={llm_pp_size}",
                ]

                command = multinode_command + [
                    f"--llm_model_name={llm_model_name}",
                ]

                if vision_model_name:
                    command.extend(
                        [
                            f"--vision_model_name={vision_model_name}",
                            f"--vision_pp_size={vision_pp_size}",
                        ]
                    )
                if audio_model_name:
                    command.extend(
                        [
                            f"--audio_model_name={audio_model_name}",
                            f"--audio_pp_size={audio_pp_size}",
                        ]
                    )

                print(f"Running: {' '.join(command)}")
                time.sleep(5)
                result = subprocess.run(command)
                if result.returncode > 0:
                    print(f"Error running command: {' '.join(command)}")

            dist.barrier()

        dist.destroy_process_group()
