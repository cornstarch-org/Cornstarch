from __future__ import annotations

import functools
from pathlib import Path

import torch
import tyro
from peft import LoraConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .common import (
    DTYPE,
    IMAGE_TOKEN,
    clip_vision_sequence_length,
    configure_special_tokens,
    expand_modality_tokens,
    generate_random_image,
    layer_offload_config,
    optional_torch_profiler,
    tokenize_text_batch,
    vision_config_from_pretrained,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    CornstarchLanguageModel,
    CornstarchModalityEncoder,
    FinetuningMode,
    RepeatedLayerCompileConfig,
    build_modality_encoder,
    configure_finetuning,
    from_hf_config,
)


class FakeDataset(Dataset):
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image = generate_random_image(image_size)
        # Keep the example below the tiny default LLM's context window after the
        # single placeholder expands to one token per vision feature row.
        self.text = IMAGE_TOKEN + " text" * 128

    def __len__(self) -> int:
        return 65536

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        return {"image": self.image, "text": self.text}


def _collate_vlm(
    batches: list[dict[str, object]],
    image_processor: BaseImageProcessor,
    tokenizer: PreTrainedTokenizerBase,
    image_sequence_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = [batch["image"] for batch in batches]
    texts = [
        expand_modality_tokens(batch["text"], {IMAGE_TOKEN: image_sequence_length})
        for batch in batches
    ]

    vision_inputs = image_processor(images=images, return_tensors="pt")
    language_inputs = tokenize_text_batch(texts, tokenizer, device)

    return {
        "pixel_values": vision_inputs["pixel_values"].to(device=device, dtype=DTYPE),
        **language_inputs,
    }


def _training_step(
    language_model: CornstarchLanguageModel,
    vision_module: CornstarchModalityEncoder,
    batch: dict[str, torch.Tensor],
    image_token_id: int,
) -> CausalLMOutputWithPast:
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=vision_module,
        pixel_values=batch["pixel_values"],
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        modality_token_ids={"vision": image_token_id},
        encoder_outputs={"vision": vision_outputs},
    )
    language_outputs = plan.run_language_model(module=language_model, inputs=merged)
    return language_outputs.execute()


def _lora_config(mode: FinetuningMode) -> LoraConfig | None:
    if mode != "lora":
        return None
    return LoraConfig(
        target_modules="all-linear",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )


def pretrain(
    vision_encoder_name_or_path: str = "openai/clip-vit-base-patch32",
    llm_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    use_layer_offload: bool = False,
    vision_train_mode: FinetuningMode = "full",
    llm_train_mode: FinetuningMode = "full",
    profile_output_path: Path | None = None,
    max_train_steps: int = 10,
) -> None:
    """Randomly initialize a VLM and train it through Cornstarch's DAG API."""
    if not torch.cuda.is_available():
        raise RuntimeError("This training example requires a CUDA GPU.")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    print(f"Pretraining a VLM with {vision_encoder_name_or_path} + {llm_name_or_path}.")

    vision_config = vision_config_from_pretrained(vision_encoder_name_or_path)
    language_root_config = AutoConfig.from_pretrained(llm_name_or_path)
    language_config = getattr(language_root_config, "text_config", language_root_config)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
    token_ids = configure_special_tokens(
        language_config, tokenizer, [IMAGE_TOKEN]
    )
    image_token_id = token_ids[IMAGE_TOKEN]
    offload_config = layer_offload_config(use_layer_offload, device)
    compile_config = RepeatedLayerCompileConfig(enabled=False)

    language_model = from_hf_config(
        language_config,
        model_kind="language",
        layer_offload_config=offload_config,
        layer_compile_config=compile_config,
    )
    vision_encoder = from_hf_config(
        vision_config,
        model_kind="vision",
        layer_offload_config=offload_config,
        layer_compile_config=compile_config,
    )
    vision_module = build_modality_encoder(
        vision_encoder,
        language_model,
        modality="vision",
    )

    language_model.set_random_init()
    vision_module.set_random_init()
    configure_finetuning(
        vision_module,
        vision_train_mode,
        lora_config=_lora_config(vision_train_mode),
    )
    configure_finetuning(
        language_model,
        llm_train_mode,
        lora_config=_lora_config(llm_train_mode),
    )
    language_model.materialize(device, dtype=DTYPE)
    vision_module.materialize(device, dtype=DTYPE)
    language_model.train()
    vision_module.train()

    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_name_or_path)
    image_sequence_length = clip_vision_sequence_length(vision_encoder.config)

    dataset = FakeDataset(image_size=(720, 480))
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=functools.partial(
            _collate_vlm,
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_sequence_length=image_sequence_length,
            device=device,
        ),
    )

    optimizer = Adam(
        param
        for param in (
            list(language_model.parameters()) + list(vision_module.parameters())
        )
        if param.requires_grad
    )
    optimizer.zero_grad()

    total_steps = min(len(dataloader), max_train_steps)
    num_warmup_steps = int(total_steps * 0.1)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    dataloader_iter = iter(dataloader)
    with optional_torch_profiler(profile_output_path) as profiler:
        with tqdm(range(total_steps)) as pbar:
            for _ in pbar:
                batch = next(dataloader_iter)
                outputs = _training_step(
                    language_model=language_model,
                    vision_module=vision_module,
                    batch=batch,
                    image_token_id=image_token_id,
                )
                loss = outputs.loss
                loss.backward()
                pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if profiler is not None:
                    profiler.step()


if __name__ == "__main__":
    tyro.cli(pretrain)
