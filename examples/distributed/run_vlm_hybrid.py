"""
An example of pretraining a vision language model (VLM) using hybrid parallelism.

Run:
torchrun (dist configs) run_vlm_ddp.py --vision-encoder_name <vision_encoder_name> --llm-name <llm_name> [--llava-dataset-file-path <llava_dataset_file_path>]

For single-node multi-GPU training: torchrun --standalone --nproc-per-node=N
For multi-node training: torchrun --master-addr <ip> --master-port <port> --nproc-per-node=N
(need to run on all nodes)
note that total number of GPUs used should be multiple of 4 to run the default example,
as it is implemented to use 2 tensor parallel degrees and 2 pipeline parallel degrees.

--llava-dataset-file-path is optional. If not given, a fake dataset will be used.
"""

import functools
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
import tyro
from colossalai.booster import Booster
from datasets import load_dataset
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin import (
    ModalParallelPlugin,
    MultimodalParallelPlugin,
)

sys.path.append(Path(__file__).parent.joinpath("..").as_posix())
from commons import (
    collate_fn,
    collate_fn_llava_pretrain,
    model_names,
    vision_encoder_classes,
)
from fake_dataset import FakeDataset


def pretrain(
    vision_encoder_name: Literal["clip", "siglip", "pixtral", "qwen2_vision"],
    llm_name: Literal["gemma2", "llama", "phi3", "mistral"],
    llava_dataset_file_path: Optional[Path] = None,
):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")

    vision_encoder_path = model_names[vision_encoder_name]
    llm_path = model_names[llm_name]
    print(f"Pretraining a VLM with {vision_encoder_path} + {llm_path}.")

    # Create a model
    with torch.device("meta"):
        vision_config = AutoConfig.from_pretrained(vision_encoder_path)
        vision_encoder = vision_encoder_classes[vision_encoder_name](
            vision_config.vision_config
        )

        llm_config = AutoConfig.from_pretrained(llm_path)
        language_model = AutoModelForCausalLM.from_config(llm_config)

        if vision_encoder_name == "qwen2_vision":
            # Qwen2 vision encoder is not designed as a standalone model,
            # but used in Qwen2VL, which has a preprocessing procedure for the input.
            # pixel_values -> hidden_states, image_grid_thw -> grid_thw
            vision_encoder = ModalEncoderModule(
                model=vision_encoder,
                additional_args=["pixel_values", "image_grid_thw"],
                preprocess_callback=lambda inputs: {
                    "hidden_states": inputs["pixel_values"],
                    "grid_thw": inputs["image_grid_thw"],
                },
            )
        else:
            vision_encoder = ModalEncoderModule(vision_encoder)

        model = MultimodalModel(
            encoders={"vision": vision_encoder},
            language_model=language_model,
        ).to(dtype=torch.bfloat16)

    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()

    # Create a processor
    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    processor = MultimodalProcessor(
        encoder_processors={"vision": image_processor},
        llm_tokenizer=tokenizer,
        model=model,
        predefined_tokens={"vision": "<image>"},
    )

    if llava_dataset_file_path:
        dataset = load_dataset("json", data_files=llava_dataset_file_path.as_posix())[
            "train"
        ]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4,
            shuffle=True,
            drop_last=True,
            collate_fn=functools.partial(
                collate_fn_llava_pretrain,
                processor=processor,
                dataset_dir=llava_dataset_file_path.parent,
            ),
        )
    else:
        print("No dataset is provided. Using a fake data iterator.")
        dataset = FakeDataset(image_size=(720, 480))
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4,
            collate_fn=functools.partial(collate_fn, processor=processor),
        )

    # Parallelize the model
    vision_encoder_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            model_name=PipelineTemplate.get_model_name(vision_encoder.module),
            modules_per_stage=[PipelineTemplate.get_modules(vision_encoder)],
        ),
    )
    language_model_plugin = ModalParallelPlugin(
        tp_size=2,
        pipeline_template=PipelineTemplate(
            model_name=PipelineTemplate.get_model_name(language_model),
            modules_per_stage=[PipelineTemplate.get_modules(language_model)],
        ),
    )
    num_microbatches = 8
    microbatch_size = 2
    plugin = MultimodalParallelPlugin(
        encoder_plugins={"vision": vision_encoder_plugin},
        language_model_plugin=language_model_plugin,
        num_microbatches=num_microbatches,
        microbatch_size=microbatch_size,
        precision=None,  # Don't use mixed precision and train with bf16
        enable_flash_attention=True,
    )
    booster = Booster(plugin=plugin)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5, fused=True
    )

    total_steps = len(dataloader)
    num_warmup_steps = int(total_steps * 0.1)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, criterion, *_ = booster.booster(
        model=model,
        optimizer=optimizer,
        criterion=lambda outputs, inputs: outputs.loss,
    )

    optimizer.zero_grad()
    is_pp_last_stage = plugin.stage_manager.is_last_stage(check_only_in_modal=False)

    dataloader_iter = iter(dataloader)
    with tqdm(
        range(total_steps),
        disable=not is_pp_last_stage,
    ) as pbar:
        for item in pbar:
            outputs = booster.execute_pipeline(
                data_iter=dataloader_iter,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                return_loss=True,
                return_outputs=False,
            )

            if is_pp_last_stage:
                loss = outputs["loss"]
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
