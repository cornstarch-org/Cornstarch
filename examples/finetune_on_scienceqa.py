import functools
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# import wandb
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoImageProcessor, get_linear_schedule_with_warmup

from transformers.models.siglip import SiglipVisionModel
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2TokenizerFast
from peft import LoraConfig, get_peft_model

from commons import collate_fn_scienceqa_pretrain

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)


def find_all_linear_names(model, component=None, exclude_component=None):
    cls = torch.nn.Linear
    identity = torch.nn.Identity
    lora_module_names = set()
    exclude_module_names = set()

    for name, module in model.named_modules():
        if component not in name:
            continue

        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    print("lora_module_names:", lora_module_names)
    lora_module_names -= exclude_module_names

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def print_param_count(model):
    print("+" * 100)
    total_trainable_params = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    total_params = sum([p.numel() for p in model.parameters()])

    ve_trainable_params = sum(
        [p.numel() for p in model.vision_encoder.parameters() if p.requires_grad]
    )
    ve_params = sum([p.numel() for p in model.vision_encoder.parameters()])

    llm_trainable_params = sum(
        [p.numel() for p in model.language_model.parameters() if p.requires_grad]
    )
    llm_params = sum([p.numel() for p in model.language_model.parameters()])

    print(
        f"[Total] trainable params: {total_trainable_params} || all params: {total_params} || trainable%: {total_trainable_params/total_params*100:.2f}"
    )
    print(
        f"[VE] trainable params: {ve_trainable_params} || all params: {ve_params} || trainable%: {ve_trainable_params/ve_params*100:.2f}"
    )
    print(
        f"[LLM] trainable params: {llm_trainable_params} || all params: {llm_params} || trainable%: {llm_trainable_params/llm_params*100:.2f}"
    )
    print("+" * 100)


def finetune(
    num_epoch: Optional[int] = 1,
    lr_ve: Optional[float] = 2e-6,
    lr_llm: Optional[float] = 1e-5,
    rank_ve: Optional[int] = -1,
    rank_llm: Optional[int] = -1,
    model_size: str = "7B",
    is_alignment: Optional[bool] = False,
    dataset_dir: Optional[Path] = None,
    dataset_file_name: Optional[str] = None,
    warmup_fraction: float = 0.1,
):
    rank_ve = "f" if rank_ve == -1 else rank_ve
    rank_llm = "f" if rank_llm == -1 else rank_llm

    ckpt_save_dir = f"/workspace/Cornstarch/examples/LLaVA-OV-{model_size}-SciQA"

    Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)

    # Create a model
    vision_encoder = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    language_model = Qwen2ForCausalLM.from_pretrained(
        f"Qwen/Qwen2-{model_size}",
    )
    model = MultimodalModel(
        encoders={"vision": ModalEncoderModule(vision_encoder)},
        language_model=language_model,
        init_projector_type="mlp",
        init_activation="gelu",
    ).to(dtype=torch.bfloat16, device="cuda")

    # Create a processor
    image_processor = AutoImageProcessor.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    text_processor = Qwen2TokenizerFast.from_pretrained(f"Qwen/Qwen2-{model_size}")
    processor = MultimodalProcessor(
        llm_tokenizer=text_processor,
        encoder_processors={"vision": image_processor},
        model=model,
        predefined_tokens={"vision": "<image>"},
    )

    # Load pretrained projector
    ckpt = torch.load(
        f"/workspace/mllm/ckpt/LLaVA-OV-{model_size}-Aligned/aligned.pt",
        map_location="cuda",
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[info] Projector checkpoint loaded")

    # Create a PEFT configuration for LLM
    peft_config = LoraConfig(
        r=rank_llm,
        lora_alpha=rank_llm * 2,
        target_modules=find_all_linear_names(model, component="language_model"),
        task_type="CAUSAL_LM",
        use_rslora=True,
        bias="none",
        init_lora_weights="gaussian",
    )
    peft_model = get_peft_model(model.language_model, peft_config)
    print_param_count(model)

    if rank_ve != "f":
        rank_ve = int(rank_ve)
        peft_config = LoraConfig(
            r=rank_ve,
            lora_alpha=rank_ve * 2,
            target_modules=find_all_linear_names(model, component="vision_model"),
            use_rslora=True,
            bias="none",
            init_lora_weights="gaussian",
        )
        peft_model = get_peft_model(model.vision_encoder.module, peft_config)
        print_param_count(model)

        peft_config = LoraConfig(
            r=rank_ve,
            lora_alpha=rank_ve * 2,
            target_modules=find_all_linear_names(model, component="projector"),
            use_rslora=True,
            bias="none",
            init_lora_weights="gaussian",
        )
        peft_model = get_peft_model(model.vision_encoder.projector, peft_config)
        print_param_count(model)

    model.gradient_checkpointing_enable()
    model.train()

    # For alignment, freeze the vision encoder and language model
    # if is_alignment:
    #     for name, param in vision_encoder.named_parameters():
    #         param.requires_grad = False

    #     for name, param in language_model.named_parameters():
    #         param.requires_grad = False

    # For ScineceQA
    dataset_train = load_dataset(
        "json", data_files=f"{dataset_dir}/{dataset_file_name}"
    )["train"]

    dataset_val = load_dataset(
        "json", data_files=f"{dataset_dir}/llava_val_QCM-LEA.json"
    )["train"]
    dataset = concatenate_datasets([dataset_train, dataset_val])
    dataset = dataset.filter(lambda example: example["image"] is not None)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,  # todo: use max bz
        shuffle=True,  # todo: True
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_scienceqa_pretrain, processor=processor, dataset_dir=dataset_dir
        ),
    )
    print(f"[info] dataset loaded {len(dataloader)}")

    optimizer_ve = Adam(model.vision_encoder.parameters(), lr=lr_ve)
    optimizer_llm = Adam(model.language_model.parameters(), lr=lr_llm)
    optimizer_ve.zero_grad()
    optimizer_llm.zero_grad()

    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_fraction)
    lr_scheduler_ve: CosineAnnealingLR = get_cosine_schedule_with_warmup(
        optimizer_ve,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    lr_scheduler_llm: CosineAnnealingLR = get_cosine_schedule_with_warmup(
        optimizer_llm,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # wandb.init(
    #     project="mars",
    #     config={
    #         "rank_ve": rank_ve,
    #         "rank_llm": rank_llm,
    #         "lr_ve": lr_ve,
    #         "lr_llm": lr_llm,
    #         "architecture": "siglip_qwen2",
    #         "dataset": "llava_instruct_150k",
    #         "epochs": num_epoch,
    #     },
    # )

    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "loss": 0,
    }
    checkpoint_path = f"{ckpt_save_dir}/before_ft.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[info] starting checkpoint saved at {checkpoint_path}")

    for epoch in range(num_epoch):

        total_step = len(dataloader)
        dataload_iter = iter(dataloader)
        iteration = 0
        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()

                optimizer_ve.step()
                optimizer_llm.step()

                lr_scheduler_ve.step()
                lr_scheduler_llm.step()

                optimizer_ve.zero_grad()
                optimizer_llm.zero_grad()

                # wandb.log(
                #     {
                #         "loss": loss.item(),
                #         "lr_ve": optimizer_ve.param_groups[0]["lr"],
                #         "lr_llm": optimizer_llm.param_groups[0]["lr"],
                #     }
                # )

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "lr_ve": optimizer_ve.param_groups[0]["lr"],
                        "lr_llm": optimizer_llm.param_groups[0]["lr"],
                    }
                )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ve_optimizer_state_dict": optimizer_ve.state_dict(),
            "llm_optimizer_state_dict": optimizer_llm.state_dict(),
            "loss": loss.item(),
        }
        checkpoint_path = f"{ckpt_save_dir}/ep{epoch + 1}_finetuned.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"[info] Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    tyro.cli(finetune)
    # wandb.finish()
