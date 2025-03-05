import functools
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from commons import (
    collate_fn,
    collate_fn_llava_pretrain,
    model_names,
    vision_encoder_classes,
)
from datasets import load_dataset
from fake_dataset import FakeDataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import Adam
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


def pretrain(
    vision_encoder_name: Literal["clip", "siglip", "pixtral", "qwen2_vision"],
    llm_name_or_path: str,
    llava_dataset_file_path: Optional[Path] = None,
):
    """
    Randomly initialize the model and pretrain it on the LLaVA-Pretrain dataset.
    """
    torch.cuda.set_device(0)

    vision_encoder_path = model_names[vision_encoder_name]
    print(f"Pretraining a VLM with {vision_encoder_path} + {llm_name_or_path}.")

    # Create a model
    with torch.device("meta"):
        vision_config = AutoConfig.from_pretrained(vision_encoder_path)
        vision_encoder = vision_encoder_classes[vision_encoder_name](
            vision_config.vision_config
        )

        llm_config = AutoConfig.from_pretrained(llm_name_or_path)
        language_model = AutoModelForCausalLM.from_config(llm_config)

        # add lora adaptors to language model
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False)
        language_model = get_peft_model(language_model, peft_config)

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

    model.gradient_checkpointing_enable()
    model.train(encoders_mode={"vision": (False, True)}, llm_mode=None)

    # materialize the model
    model.to_empty(device="cuda")

    # Create a processor
    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    processor = MultimodalProcessor(
        encoder_processors={"vision": image_processor},
        llm_tokenizer=tokenizer,
        model=model,
        # by default, vision encoder uses "<vision>" as its special token,
        # while Llava pretrain dataset of fake dataset includes "<image>".
        # So, we need to specify the token for the vision encoder.
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
    optimizer = Adam([p for p in model.parameters() if p.requires_grad])
    optimizer.zero_grad()

    total_steps = len(dataloader)
    num_warmup_steps = int(total_steps * 0.1)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    dataloader_iter = iter(dataloader)
    with tqdm(
        range(total_steps),
    ) as pbar:
        for item in pbar:
            inputs = next(dataloader_iter)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    tyro.cli(pretrain)
