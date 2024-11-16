import csv
import os
import re
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.lazy import LazyInitContext
from torch.optim import Optimizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
)
from cornstarch.pipeline_template import PipelineTemplate
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
from cornstarch.testing.timer import TimerContextManager

file_path = "profile_pipeline_result.csv"

model_names_to_test = [
    ("llama_8b", "clip", "qwen2_audio"),
]


class CornstarchTestingClass:
    num_microbatches: int = 16
    microbatch_size: int = 1

    def __init__(
        self,
        llm_model_class: LanguageModelClassBase,
        encoder_model_classes: dict[str, ImageModelClassBase | AudioModelClassBase],
    ):
        llm_model_class.config.num_hidden_layers = 2
        for encoder_class in encoder_model_classes.values():
            encoder_class.config.num_hidden_layers = 1
        self.llm_model_class = llm_model_class
        self.encoder_model_classes = encoder_model_classes

    def data(self) -> dict[str, torch.Tensor]:
        data = {}
        batch_size = self.num_microbatches * self.microbatch_size
        data.update(self.llm_model_class.data(batch_size, seq_len=4096))

        if "vision" in self.encoder_model_classes:
            data.update(self.encoder_model_classes["vision"].data(batch_size))
        if "audio" in self.encoder_model_classes:
            data.update(self.encoder_model_classes["audio"].data(batch_size))

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

        with LazyInitContext():
            encoders = {
                key: ModalEncoderModule(encoder_class.build_model())
                for key, encoder_class in self.encoder_model_classes.items()
            }
            for encoder in encoders.values():
                encoder.train(module=False, projector=True)
                encoder.module.requires_grad_(False)
                if encoder.projector is not None:
                    encoder.projector.requires_grad_(True)

            model = MultimodalModel(
                encoders=encoders,
                language_model=self.llm_model_class.build_model(),
            )

        model.language_model.train(mode=False)
        model.language_model.requires_grad_(False)
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

        plugin = MultimodalParallelPlugin(
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
        return model, optimizer, criterion, booster

    def run_multimodal_model(
        self,
        model: MultimodalParallelModule,
        optimizer: Optimizer,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        booster: Booster,
    ):
        num_iterations = 5

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        data = self.data()
        data_iter = iter([data for _ in range(num_iterations)])

        for _ in range(num_iterations):
            booster.execute_pipeline(
                data_iter,
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

    dist.barrier()

    manager = TimerContextManager()

    with manager.measure(
        "cornstarch.plugin.multimodal_parallel_plugin.multimodal_1f1b.MultimodalEncoderTrainingOneForwardOneBackwardSchedule.run_forward_backward"
    ):
        cornstarch_class.run_multimodal_model(model, optimizer, criterion, booster)

        torch.cuda.synchronize()

    if dist.get_rank() == 0:
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
                }
            )

    dist.barrier()
    dist.destroy_process_group()


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

        if args.vision_pp_size > 0:
            assert args.vision_model_name is not None
            kargs.update(
                {
                    "vision_model_name": args.vision_model_name,
                    "vision_pp_size": args.vision_pp_size,
                }
            )
        if args.audio_pp_size > 0:
            assert args.audio_model_name is not None
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
        import subprocess
        import sys

        # If LOCAL_RANK is not set, we are in the main process and need to launch child processes
        num_gpus = 4  # Set this to the number of GPUs you want to use

        for llm_model_name, vision_model_name, audio_model_name in model_names_to_test:
            command = [
                "torchrun",
                "--nproc_per_node",
                str(num_gpus),
                sys.argv[0],  # The current script file
                "--llm_model_name",
                llm_model_name,
                "--llm_pp_size",
                "2",
                "--tp_size",
                "1",
            ]

            if vision_model_name:
                command.extend(
                    [
                        "--vision_model_name",
                        vision_model_name,
                        "--vision_pp_size",
                        "1",
                    ]
                )
            if audio_model_name:
                command.extend(
                    [
                        "--audio_model_name",
                        audio_model_name,
                        "--audio_pp_size",
                        "1",
                    ]
                )

            print(f"Running: {' '.join(command)}")
            subprocess.run(command)
