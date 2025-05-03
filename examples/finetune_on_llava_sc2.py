import functools
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro
from datasets import load_dataset, concatenate_datasets
from PIL import Image
from torch.optim import Adam

# from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers import AutoImageProcessor

from transformers.models.siglip import SiglipVisionModel
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2TokenizerFast
from peft import LoraConfig, get_peft_model

from commons import collate_fn_llava_pretrain

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)

use_wandb = False


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

    lora_module_names -= exclude_module_names

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")

    print("lora_module_names:", lora_module_names)

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


def calculate_dataset_perplexity_trainset(model, dataloader_val, device: str = "cuda"):
    """
    Calculate perplexity over a dataset using multiple batches.

    Args:
        model: The causal LM model.
        dataloader_val: DataLoader with (image, labels) pairs or just labels.
        vocab_size: Size of the vocabulary.
        device: Device to run the computation on.

    Returns:
        float: Dataset-level perplexity.
    """
    total_step = len(dataloader_val)
    print(f"total_step: {total_step}")
    dataload_iter = iter(dataloader_val)

    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(
            range(total_step),
            desc=f"Val on Trainset",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                total_loss += outputs.loss

    avg_nll = total_loss / total_step
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return avg_nll, perplexity


def calculate_dataset_perplexity_testset(model, dataloader_val, device: str = "cuda"):
    """
    Calculate perplexity over a dataset using multiple batches.

    Args:
        model: The causal LM model.
        dataloader_val: DataLoader with (image, labels) pairs or just labels.
        vocab_size: Size of the vocabulary.
        device: Device to run the computation on.

    Returns:
        float: Dataset-level perplexity.
    """
    total_step = len(dataloader_val)
    print(f"total_step: {total_step}")
    dataload_iter = iter(dataloader_val)

    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(
            range(total_step),
            desc=f"Val on Testset",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                total_loss += outputs.loss

    avg_nll = total_loss / total_step
    perplexity = torch.exp(torch.tensor(avg_nll)).item()

    return avg_nll, perplexity


def finetune(
    num_epoch: Optional[int] = 100,
    lr_ve: Optional[float] = 1e-5,  # 2e-6,
    lr_llm: Optional[float] = 1e-5,  # 1e-5,
    rank_ve: Optional[int] = 4,
    rank_llm: Optional[int] = 32,
    data_scale: Optional[int] = 0,
    model_size: str = "7B",
    is_alignment: Optional[bool] = False,
    dataset_dir: Optional[Path] = None,
    warmup_fraction: float = 0.1,
):
    rank_ve = "f" if rank_ve == -1 else rank_ve
    rank_llm = "f" if rank_llm == -1 else rank_llm
    data_amount = 2 ** (data_scale) if data_scale >= 0 else "full"

    ckpt_save_dir = f"/workspace/Cornstarch/examples/SC2-{model_size}-rVE{rank_ve}_rLLM{rank_llm}_{data_amount}_samelr_1e5"  # put your save_dir path here

    Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    trainlog_file = f"{ckpt_save_dir}/trainlog.log"
    f_trainlog = open(trainlog_file, "w")
    vallog_file = f"{ckpt_save_dir}/vallog.log"
    f_vallog = open(vallog_file, "w")

    # Create a model
    vision_encoder = SiglipVisionModel.from_pretrained(
        "google/siglip-so400m-patch14-384"
    )
    language_model = Qwen2ForCausalLM.from_pretrained(
        f"Qwen/Qwen2-{model_size}-Instruct",
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
    text_processor = Qwen2TokenizerFast.from_pretrained(
        f"Qwen/Qwen2-{model_size}-Instruct"
    )
    processor = MultimodalProcessor(
        llm_tokenizer=text_processor,
        encoder_processors={"vision": image_processor},
        model=model,
        predefined_tokens={"vision": "<image>"},
    )

    # Load pretrained projector
    ckpt = torch.load(
        f"/workspace/mllm/ckpt/LLaVA-OV-{model_size}-Align-Instruct/aligned.pt",  # put your checkpoint path here
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

    """
    Examples of loading some datasets:
    1. liuhaotian/llava-pretrain
        tree -L 1 /path/to/datasets/liuhaotian___llava-pretrain
        liuhaotian___llava-pretrain/
        |-- 00000
        |-- 00001
        |-- 00002
        ...
        |-- 00658
        |-- 00659
        |-- blip_laion_cc_sbu_558k.json

        dataset_dir = "/workspace/mllm/dataset/LCS/liuhaotian___llava-pretrain"
        dataset_file_name = "blip_laion_cc_sbu_558k.json"
    """
    dataset_dir = "/workspace/mllm/dataset/LLaVA-mixture/coco/train2017"  # put your dataset path here
    traindata_file_name = (
        f"llava_instruct_smalltrain_{data_amount}.json"
        if data_scale >= 0
        else "llava_instruct_train_142k.json"
    )
    minitrain_file_name = "llava_instruct_smalltrain_512.json"
    valdata_file_name = "llava_instruct_smalltval_512.json"
    dataset = load_dataset("json", data_files=f"{dataset_dir}/{traindata_file_name}")[
        "train"
    ]
    dataset_minitrain = load_dataset(
        "json", data_files=f"{dataset_dir}/{minitrain_file_name}"
    )["train"]
    dataset_val = load_dataset("json", data_files=f"{dataset_dir}/{valdata_file_name}")[
        "train"
    ]

    batch_size = 8
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # todo: use max bz
        shuffle=False,  # todo: True
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain, processor=processor, dataset_dir=dataset_dir
        ),
    )
    dataloader_minitrain = DataLoader(
        dataset=dataset_minitrain,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain, processor=processor, dataset_dir=dataset_dir
        ),
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=functools.partial(
            collate_fn_llava_pretrain, processor=processor, dataset_dir=dataset_dir
        ),
    )

    print(f"[info] dataset loaded {len(dataloader)}")

    optimizer_ve = Adam(model.vision_encoder.parameters(), lr=lr_ve)
    optimizer_llm = Adam(model.language_model.parameters(), lr=lr_llm)
    optimizer_ve.zero_grad()
    optimizer_llm.zero_grad()

    total_steps = len(dataloader) * num_epoch
    num_warmup_steps = int(total_steps * warmup_fraction)
    # lr_scheduler_ve: CosineAnnealingLR = get_cosine_schedule_with_warmup(
    #     optimizer_ve,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps,
    # )
    # lr_scheduler_llm: CosineAnnealingLR = get_cosine_schedule_with_warmup(
    #     optimizer_llm,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps,
    # )
    if use_wandb:
        wandb.init(
            project="mars_march25",
            config={
                "rank_ve": rank_ve,
                "rank_llm": rank_llm,
                "lr_ve": lr_ve,
                "lr_llm": lr_llm,
                "architecture": "siglip_qwen2",
                "dataset": "llava_instruct_150k",
                "epochs": num_epoch,
            },
        )

    # checkpoint = {
    #     "epoch": 0,
    #     "model_state_dict": model.state_dict(),
    #     "loss": 0,
    # }
    # checkpoint_path = f"{ckpt_save_dir}/before_ft.pt"
    # torch.save(checkpoint, checkpoint_path)
    # print(f"[info] starting checkpoint saved at {checkpoint_path}")

    iteration = 0
    curr_ppl = 0
    for epoch in range(num_epoch):
        # if epoch >= 1:
        #     break

        if epoch >= 50:
            break
        if iteration >= 10000:
            break

        if epoch > 10 and curr_ppl > 12.0:  # early stop
            break

        model.train()

        total_step = len(dataloader)
        dataload_iter = iter(dataloader)

        loss2 = 0

        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                loss2 = loss.item()

                optimizer_ve.step()
                optimizer_llm.step()

                # lr_scheduler_ve.step()
                # lr_scheduler_llm.step()

                optimizer_ve.zero_grad()
                optimizer_llm.zero_grad()

                if use_wandb:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "lr_ve": optimizer_ve.param_groups[0]["lr"],
                            "lr_llm": optimizer_llm.param_groups[0]["lr"],
                        }
                    )

                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "lr_ve": optimizer_ve.param_groups[0]["lr"],
                        "lr_llm": optimizer_llm.param_groups[0]["lr"],
                    }
                )

                if iteration % 5 == 0:
                    avg_nll_train, perplexity_train = (
                        calculate_dataset_perplexity_trainset(
                            model,
                            dataloader_minitrain,
                            device="cuda",
                        )
                    )

                    avg_nll_val, perplexity_val = calculate_dataset_perplexity_testset(
                        model,
                        dataloader_val,
                        device="cuda",
                    )
                    curr_ppl = perplexity_val

                    f_trainlog.write(
                        f"[result] Epoch {epoch} Iteration {iteration} loss: {loss2}\n"
                    )
                    f_trainlog.write(
                        f"[result] Epoch {epoch} Iteration {iteration} perplexity: {perplexity_train}\n"
                    )
                    f_vallog.write(
                        f"[result] Epoch {epoch} Iteration {iteration} loss: {loss2}\n"
                    )
                    f_vallog.write(
                        f"[result] Epoch {epoch} Iteration {iteration} perplexity: {perplexity_val}\n"
                    )

                iteration += 1

    f_trainlog.flush()
    f_trainlog.close()
    f_vallog.flush()
    f_vallog.close()


if __name__ == "__main__":
    tyro.cli(finetune)

    if use_wandb:
        wandb.finish()
