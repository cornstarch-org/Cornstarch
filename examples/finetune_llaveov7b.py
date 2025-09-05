import sys
import os

# Calculate the project root directory (assuming this script is in Cornstarch/examples/)
# and add it to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import functools
from pathlib import Path
from typing import Literal, Optional

import torch
import tyro 
import wandb
from commons import (
    collate_fn_scienceqa,
    collate_fn_scienceqa_eval_single,
    collate_fn_scienceqa_llavaovhf,
    collate_fn_scienceqa_eval_single_llavaovhf,
)
from datasets import load_dataset, concatenate_datasets
from fake_dataset import FakeDataset
from peft import LoraConfig, TaskType, get_peft_model

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, get_cosine_schedule_with_warmup
from qwen_vl_utils import process_vision_info

from Cornstarch.examples.scienceqa_loader import preprocess_local_scienceqa, raw_dataset_list, BASE_DIR
from Cornstarch.examples.scienceqa_judge import calculate_accuracy

use_wandb = False

def find_all_linear_names(model, component=None, exclude_component=None):
    cls = torch.nn.Linear
    identity = torch.nn.Identity
    lora_module_names = set()
    exclude_module_names = set()

    for name, module in model.named_modules():
        if component is not None and component not in name:
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
        [p.numel() for p in model.vision_tower.parameters() if p.requires_grad]
    )
    ve_params = sum([p.numel() for p in model.vision_tower.parameters()])

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


def calculate_dataset_perplexity(model, dataloader_test, device: str = "cuda"):
    """
    Calculate perplexity over a dataset using multiple batches.

    Args:
        model: The causal LM model.
        dataloader_test: DataLoader with (image, labels) pairs or just labels.
        vocab_size: Size of the vocabulary.
        device: Device to run the computation on.

    Returns:
        float: Dataset-level perplexity.
    """
    total_step = len(dataloader_test)
    print(f"total_step: {total_step}")
    dataload_iter = iter(dataloader_test)

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

    # generate answer by decoding outputs
    model.generate(inputs["input_ids"], max_new_tokens=100)
    print (f"outputs: {outputs}")
    
    avg_nll = total_loss / total_step
    perplexity = torch.exp(torch.tensor(avg_nll)).item()
    return avg_nll, perplexity


def evaluate_scienceqa_accuracy(model, dataloader_test, processor, device: str = "cuda", f_log: open = None, epoch: int = 0):
    """
    Calculate accuracy on the ScienceQA dataset.

    Args:
        model: The trained Qwen2.5_VL model.
        dataloader_test: DataLoader yielding batches. Each batch MUST be a dict
                         containing 'model_inputs' (dict with 'input_ids', etc.),
                         'ground_truths' (list of str), and 'choices' (list of lists).
        processor: The Qwen2VLProcessor for decoding.
        device: Device to run the computation on.

    Returns:
        float: Dataset-level accuracy.
    """
    total_step = len(dataloader_test)
    print(f"Total steps for evaluation: {total_step}")
    if total_step == 0:
        print("DataLoader is empty, cannot evaluate.")
        return 0.0
        
    dataload_iter = iter(dataloader_test)

    all_generated_texts = []
    all_ground_truths = []
    all_choices_list = []

    model.eval()
    model.to(device)

    with torch.no_grad():
        with tqdm(
            range(total_step),
            desc=f"Evaluating ScienceQA Accuracy",
        ) as pbar:
            for item_idx in pbar:
                if item_idx > 500:
                    break
                try:
                    # --- A. Get Batch and Check Structure ---
                    batch = next(dataload_iter)
                    
                    if not all(k in batch for k in ['model_inputs', 'ground_truths', 'choices']):
                        raise ValueError("Batch structure incorrect. Expected 'model_inputs', 'ground_truths', 'choices'.")

                    model_inputs = batch['model_inputs']
                    ground_truths = batch['ground_truths']
                    choices = batch['choices']

                    # Move model inputs to the correct device
                    model_inputs = {k: v.to(device) for k, v in model_inputs.items() if isinstance(v, torch.Tensor)}

                    # --- B. Generate Text ---
                    generated_ids = model.generate(
                        # input_ids=model_inputs["input_ids"],
                        # attention_mask=model_inputs["attention_mask"],
                        # pixel_values=model_inputs.get("pixel_values"),
                        max_new_tokens=100,  # Adjust length as needed
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        do_sample=False,
                        num_beams=1,
                        **model_inputs,
                    )


                    # --- C. Decode Generated Text ---
                    input_len = model_inputs["input_ids"].shape[1]
                    only_generated_ids = generated_ids[:, input_len:]
                    batch_generated_texts = processor.batch_decode(
                        only_generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    # --- D. Store Results ---
                    all_generated_texts.extend(batch_generated_texts)
                    all_ground_truths.extend(ground_truths)
                    all_choices_list.extend(choices)
                    
                    pbar.set_postfix({"Processed": len(all_generated_texts)})

                    del model_inputs
                    del generated_ids
                    del only_generated_ids
                    torch.cuda.empty_cache()

                except StopIteration:
                    print("DataLoader finished.")
                    break
              

    # --- E. Calculate Final Accuracy ---
    if not all_generated_texts:
        print("No texts were generated, cannot calculate accuracy.")
        return 0.0
        
    total_count, correct_count, undetermined_count, accuracy = calculate_accuracy(
        all_generated_texts,
        all_ground_truths,
        all_choices_list
    )

    # output total_count, correct_count, undetermined_count, accuracy to file
    f_log.write(f"[result] Epoch {epoch} Total Count: {total_count}\n")
    f_log.write(f"[result] Epoch {epoch} Correct Count: {correct_count}\n")
    f_log.write(f"[result] Epoch {epoch} Undetermined Count: {undetermined_count}\n")
    f_log.write(f"[result] Epoch {epoch} Accuracy: {accuracy}\n")
    f_log.flush()

    return accuracy


def finetune(
    num_epoch: Optional[int] = 3,
    lr_ve: Optional[float] = 5e-6,  # 2e-6,
    lr_llm: Optional[float] = 1e-5,  # 1e-5,
    rank_ve: Optional[int] = -1,
    rank_llm: Optional[int] = -1,
    data_scale: Optional[int] = 0,
    model_size: str = "7B",
    batch_size: Optional[int] = 8,
    is_alignment: Optional[bool] = False,
    dataset_dir: Optional[Path] = None,
    warmup_fraction: float = 0.05,
):    
    """
    Finetune Qwen2.5-VL on the LLaVA-Pretrain dataset.
    """
    # Here we use the default rank_ve and rank_llm values
    rank_ve = "f" if rank_ve == -1 else rank_ve
    rank_llm = "f" if rank_llm == -1 else rank_llm

    dataset_dir = Path("/workspace/mllm/dataset/ScienceQA/data/scienceqa")

    ckpt_save_dir = f"/workspace/Cornstarch/examples/LLaVA-OV-{model_size}-rVE{rank_ve}_rLLM{rank_llm}-lr{lr_llm}"  # put your save_dir path here
    Path(ckpt_save_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{ckpt_save_dir}/log.log"
    f_log = open(log_file, "w")

    config_log = (
        f"{'-'*100}\n"
        f"num_epoch: {num_epoch}\n"
        f"lr_ve: {lr_ve}\n"
        f"lr_llm: {lr_llm}\n"
        f"rank_ve: {rank_ve}\n"
        f"rank_llm: {rank_llm}\n"
        f"data_scale: {data_scale}\n"
        f"model_size: {model_size}\n"
        f"ckpt_save_dir: {ckpt_save_dir}\n"
        f"dataset_dir: {dataset_dir}\n"
        f"{'-'*100}\n"
    )
    print(config_log, end="")
    f_log.write(config_log)
    f_log.flush()

    torch.cuda.set_device(0)

    with torch.device("cuda"):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        model.to("cuda") # Ensure model is on CUDA
        processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")

        # 1. add uniform lora adaptors to entire model
        # peft_config = LoraConfig(r=16, lora_alpha=32, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj"])
        # model = get_peft_model(model, peft_config)

        # 2. add mars lora adaptors to entire model
        peft_config = LoraConfig(
            r=rank_llm,
            lora_alpha=rank_llm * 2,
            target_modules=find_all_linear_names(model.language_model, component="model"),
            task_type="CAUSAL_LM",
            bias="none",
            init_lora_weights="gaussian",
        )
        peft_model = get_peft_model(model.language_model, peft_config)
        print_param_count(model)

        # # Option 1: when freezing VE
        # for name, param in model.named_parameters():
        #     if "vision_encoder" in name:
        #         param.requires_grad = False
        # print_param_count(model)

        # Option 2: when finetuning VE
        if rank_ve != "f":
            rank_ve = int(rank_ve)
            peft_config = LoraConfig(
                r=rank_ve,
                lora_alpha=rank_ve * 2,
                target_modules=find_all_linear_names(model.vision_tower, component="encoder"),
                bias="none",
                init_lora_weights="gaussian",
            )
            peft_model = get_peft_model(model.vision_tower, peft_config)
            print_param_count(model)

            peft_config = LoraConfig(
                r=rank_ve,
                lora_alpha=rank_ve * 2,
                target_modules=find_all_linear_names(model.multi_modal_projector),
                bias="none",
                init_lora_weights="gaussian",
            )
            peft_model = get_peft_model(model.multi_modal_projector, peft_config)
            print_param_count(model)
        
    # save model
    model.gradient_checkpointing_enable()
    model.train()

    
    train_dataset = [
        processed
        for i, example in enumerate(raw_dataset_list)
        if (processed := preprocess_local_scienceqa(example, BASE_DIR, i, 'train')) is not None
    ]

    val_dataset = [
        processed
        for i, example in enumerate(raw_dataset_list)
        if (processed := preprocess_local_scienceqa(example, BASE_DIR, i, 'val', use_image=False)) is not None
    ]


    if dataset_dir:
        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda batches: collate_fn_scienceqa_llavaovhf(batches, processor),
        )
        dataloader_test = DataLoader(
            dataset=val_dataset, # NOTE: val_dataset is used for evaluation
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batches: collate_fn_scienceqa_eval_single_llavaovhf(batches, processor),
        )
    else:
        raise ValueError("dataset_dir is not defined")
        
    print(f"[info]train dataset loaded {len(train_dataset)}") # 12726
    print(f"[info]test dataset loaded {len(val_dataset)}") # 2097

    #######################################
    # rank_ve, rank_llm = mars(model, dataloader_test, processor, device="cuda")
    #######################################
    
    vision_params = list(model.vision_tower.parameters()) + list(model.multi_modal_projector.parameters())
    optimizer_ve = Adam(vision_params, lr=lr_ve)
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

    curr_step = 0
    for epoch in range(num_epoch):
        total_step = len(dataloader)
        dataload_iter = iter(dataloader)

        with tqdm(
            range(total_step),
            desc=f"Epoch [{epoch + 1}/{num_epoch}]",
        ) as pbar:
            for item in pbar:
                inputs = next(dataload_iter)
                outputs = model(**inputs)
                loss = outputs.loss

                try:
                    loss.backward()

                    curr_step += 1

                    optimizer_ve.step()
                    optimizer_llm.step()

                    lr_scheduler_ve.step()
                    lr_scheduler_llm.step()

                    optimizer_ve.zero_grad()
                    optimizer_llm.zero_grad()
                
                except RuntimeError as e:
                    if "element 0 of tensors does not require grad" in str(e):
                        print(f"\nWARNING: Skipping batch {item} due to gradient error: {e}")
                        # Ensure gradients are cleared before the next iteration
                        optimizer_ve.zero_grad()
                        optimizer_llm.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    else:
                        # Re-raise any other RuntimeError
                        raise e

                torch.cuda.empty_cache()

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
                

            # avg_nll, perplexity = calculate_dataset_perplexity(model, dataloader_test, device="cuda")
            accuracy = evaluate_scienceqa_accuracy(model, dataloader_test, processor, device="cuda", f_log=f_log, epoch=epoch + 1)

            if use_wandb:
                wandb.log(
                    {
                        "epoch_acc": accuracy,
                    }
                )

            f_log.write(f"[result] Epoch {epoch + 1} accuracy: {accuracy}\n")

        if torch.cuda.is_available():
            model.to('cpu')
            torch.cuda.empty_cache()
            model.to('cuda')
        
        model.train()

    f_log.flush()
    f_log.close()
 

if __name__ == "__main__":
    tyro.cli(finetune)