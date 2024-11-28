import csv
import math
import os
import re
import time
from typing import Any, Callable

import numpy as np
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
from cornstarch.plugin.encoders_colocated_plugin.encoders_colocated_plugin import (
    EncodersColocatedMultimodalParallelPlugin,
)
from cornstarch.plugin.encoders_replicated_plugin.encoders_replicated_plugin import (
    EncodersReplicatedMultimodalParallelPlugin,
)
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelModule,
    MultimodalParallelPlugin,
)
from cornstarch.testing.model_zoo import (
    AudioModelClassBase,
    ImageModelClassBase,
    LanguageModelClassBase,
    model_to_class,
)
from cornstarch.testing.nodes_info import master_node_rdzv_backend, node_hostnames
from cornstarch.testing.timer import TimerContextManager

file_path = "profile_4d_result.csv"

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

# for 1024 tokens
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
        plugin_type: str,
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

        for encoder in encoders.values():
            encoder.train(module=False, projector=True)
            encoder.module.requires_grad_(False)
            encoder.projector.requires_grad_(True)
            try:
                encoder.module.gradient_checkpointing_enable({"use_reentrant": False})
            except ValueError:
                pass

        model.language_model.train(mode=False)
        model.language_model.requires_grad_(False)
        model.language_model.gradient_checkpointing_enable({"use_reentrant": False})
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
            sp_size=2,
            sequence_parallelism_mode="ring_attn",
            pipeline_template=self.get_megatron_style_pipeline_template(
                model.language_model, llm_pp_size
            ),
        )

        encoders_plugins = {
            key: ModalParallelPlugin(
                tp_size=tp_size,
                sp_size=2,
                sequence_parallelism_mode="ring_attn",
                pipeline_template=self.get_megatron_style_pipeline_template(
                    model.get_submodule(f"{key}_encoder"), encoders_pp_size[key]
                ),
            )
            for key in encoders
        }

        if plugin_type == "cornstarch":
            plugin = MultimodalParallelPlugin(
                encoder_plugins=encoders_plugins,
                language_model_plugin=llm_plugin,
                **test_config,
            )
            plugin.shard_config.ring_attention_distribution_mode = "random"
        elif plugin_type == "colocated":
            plugin = EncodersColocatedMultimodalParallelPlugin(
                encoder_plugins=encoders_plugins,
                language_model_plugin=llm_plugin,
                **test_config,
            )
            plugin.shard_config.ring_attention_distribution_mode = "zigzag"
        elif plugin_type == "replicated":
            plugin = EncodersReplicatedMultimodalParallelPlugin(
                language_model_plugin=llm_plugin,
                **test_config,
            )
            plugin.shard_config.ring_attention_distribution_mode = "zigzag"
        else:
            raise ValueError(f"Unknown plugin type {plugin_type}")

        plugin.precision = None
        booster = Booster(plugin=plugin)

        def loss_fn(x: CausalLMOutputWithPast) -> torch.Tensor:
            return x.loss

        model, optimizer, criterion, *_ = booster.boost(model, optimizer, loss_fn)
        model.to(dtype=torch.bfloat16)
        return model, optimizer, criterion, booster

    def run_multimodal_model(
        self,
        model: MultimodalParallelModule,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        booster: Booster,
    ):
        num_iterations = 2

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        data = self.data()
        data_iter = iter([data for _ in range(num_iterations)])

        for i in range(num_iterations):
            print(f"Iteration {i}")
            with torch.autograd.profiler.emit_nvtx(), torch.cuda.nvtx.range(f"iter{i}"):
                booster.execute_pipeline(
                    data_iter,
                    model,
                    lambda outputs, inputs: criterion(outputs),
                    optimizer,
                    return_loss=True,
                    return_outputs=False,
                )


def run_profile(
    plugin_type: str,
    llm_model_name: str,
    llm_pp_size: int,
    tp_size: int,
    vision_model_name: str | None = None,
    audio_model_name: str | None = None,
    vision_pp_size: int | None = None,
    audio_pp_size: int | None = None,
):
    assert plugin_type in ["cornstarch", "colocated", "replicated"]

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
        plugin_type=plugin_type,
        test_config=dict(),
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.barrier(device_ids=[local_rank])

    manager = TimerContextManager()

    if plugin_type == "cornstarch":
        fb_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.run_forward_backward"
        fwd_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.forward_step"
    elif plugin_type == "colocated":
        fb_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.run_forward_backward"
        fwd_measure_name = "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.forward_step"
    elif plugin_type == "replicated":
        fb_measure_name = "colossalai.pipeline.schedule.one_f_one_b.OneForwardOneBackwardSchedule.run_forward_backward"
        fwd_measure_name = "colossalai.pipeline.schedule.one_f_one_b.OneForwardOneBackwardSchedule.forward_step"
    else:
        raise ValueError(f"Unknown plugin type {plugin_type}")

    with manager.measure([fb_measure_name, fwd_measure_name]):
        cornstarch_class.run_multimodal_model(model, optimizer, criterion, booster)

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

    elapsed_times = manager.get_elapsed_times()

    tp_rank = dist.get_rank(booster.plugin.tp_group)
    sp_rank = dist.get_rank(booster.plugin.sp_group)
    if tp_rank == 0 and sp_rank == 0:
        try:
            pp_groups = booster.plugin.pp_groups
        except AttributeError:
            pp_groups = [booster.plugin.pp_group]

        fwd_times = elapsed_times[fwd_measure_name]
        average_fwd_time = torch.tensor(
            sum(fwd_times[1:]) / (len(fwd_times) - 1),
            device="cuda",
        )

        # Hacky way to pick one process group across all ranks. The order of pp_group in pp_groups might be different between ranks.
        pp_group_rank_sum = [
            sum(dist.get_process_group_ranks(pp_group)) for pp_group in pp_groups
        ]
        min_rank_sum_pp_group = pp_groups[
            pp_group_rank_sum.index(min(pp_group_rank_sum))
        ]

        print(
            f"Gathering forward time across pipeline stages... rank: {dist.get_rank()}, "
            f"group ranks: {dist.get_process_group_ranks(min_rank_sum_pp_group)}"
        )
        average_fwd_times = torch.empty(
            dist.get_world_size(min_rank_sum_pp_group), device="cuda"
        )
        dist.all_gather_into_tensor(
            average_fwd_times, average_fwd_time, group=min_rank_sum_pp_group
        )

        fb_times = elapsed_times[fb_measure_name]
        average_exec_time = sum(fb_times[1:]) / (len(fb_times) - 1)

        with open(
            f"pp{dist.get_rank(min_rank_sum_pp_group)}_{file_path}", "a", newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "type",
                    "llm_model",
                    "vision_model",
                    "audio_model",
                    "exec_time (ms)",
                    "fwd_time_per_pp (ms)",
                    "peak_memory (GB)",
                    "min_peak_memory (GB)",
                    "max_peak_memory (GB)",
                ],
            )
            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(
                {
                    "type": plugin_type,
                    "llm_model": llm_model_name,
                    "vision_model": vision_model_name,
                    "audio_model": audio_model_name,
                    "exec_time (ms)": average_exec_time,
                    "fwd_time_per_pp (ms)": str(average_fwd_times.tolist()),
                    "peak_memory (GB)": avg_peak_memory,
                    "min_peak_memory (GB)": min_peak_memory,
                    "max_peak_memory (GB)": max_peak_memory,
                }
            )

    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    torch.cuda.synchronize()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:

        print(
            f"Rank: {int(os.environ['RANK'])}, World size: {int(os.environ['WORLD_SIZE'])}"
        )

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
        parser.add_argument(
            "--plugin_type",
            type=str,
            default="cornstarch",
            help="Plugin type (cornstarch, colocated, replicated)",
        )
        args = parser.parse_args()

        kargs = {
            "llm_model_name": args.llm_model_name,
            "llm_pp_size": args.llm_pp_size,
            "tp_size": args.tp_size,
            "plugin_type": args.plugin_type,
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

        # for plugin_type in ["colocated"]:
        for plugin_type in ["colocated", "replicated", "cornstarch"]:
            for model_index, (
                llm_model_name,
                vision_model_name,
                audio_model_name,
            ) in enumerate(model_names_to_test):
                if vision_model_name is None and audio_model_name is None:
                    continue

                if plugin_type == "cornstarch":
                    llm_pp_size, vision_pp_size, audio_pp_size = (
                        cornstarch_parallel_configuration[model_index]
                    )
                    num_ranks = (llm_pp_size * 2 + vision_pp_size + audio_pp_size) * 2
                    if vision_model_name is None:
                        num_ranks -= vision_pp_size * 2
                    if audio_model_name is None:
                        num_ranks -= audio_pp_size * 2
                elif plugin_type == "colocated":
                    llm_pp_size, vision_pp_size, audio_pp_size = (
                        colocate_parallel_configuration[model_index]
                    )
                    assert vision_pp_size == audio_pp_size
                    num_ranks = (llm_pp_size * 2 + vision_pp_size) * 2
                else:
                    llm_pp_size, vision_pp_size, audio_pp_size = (
                        6,
                        1,
                        1,
                    )
                    num_ranks = 24

                print(
                    f"Running {llm_model_name}+{vision_model_name}+{audio_model_name} with {num_ranks} ranks"
                )

                num_nodes_to_join = math.ceil(num_ranks / 4)
                if my_node_index < len(node_hostnames) - num_nodes_to_join:
                    print(
                        "This node does not join the current model run. Waiting for the job to finish."
                    )
                else:
                    node_rank = my_node_index - (
                        len(node_hostnames) - num_nodes_to_join
                    )
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
                        f"--plugin_type={plugin_type}",
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
