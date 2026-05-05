from __future__ import annotations

import functools

import torch
import tyro
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import (
    DTYPE,
    IMAGE_TOKEN,
    build_modality_encoder,
    clip_vision_sequence_length,
    configure_special_tokens,
    expand_modality_tokens,
    generate_random_image,
    tokenize_text_batch,
    vision_config_from_pretrained,
)
from new_cornstarch.models import (
    CornstarchExecutionPlan,
    from_hf_config,
)


class FakeDataset(Dataset):
    def __init__(self, image_size: tuple[int, int]):
        self.image = generate_random_image(image_size)
        self.text = IMAGE_TOKEN + " text" * 256

    def __len__(self) -> int:
        return 65536

    def __getitem__(self, index: int) -> dict:
        del index
        return {"image": self.image, "text": self.text}


def _collate_vlm(
    batches: list[dict],
    image_processor,
    tokenizer,
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
    language_model,
    vision_module,
    batch: dict[str, torch.Tensor],
    image_token_id: int,
):
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


def pretrain(
    vision_encoder_name_or_path: str = "openai/clip-vit-base-patch32",
    llm_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
):
    """Randomly initialize a VLM and pretrain it through the new Cornstarch API."""
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print(f"Pretraining a VLM with {vision_encoder_name_or_path} + {llm_name_or_path}.")

    vision_config = vision_config_from_pretrained(vision_encoder_name_or_path)
    language_config = AutoConfig.from_pretrained(llm_name_or_path)

    language_model = from_hf_config(language_config, model_kind="language")
    vision_encoder = from_hf_config(vision_config, model_kind="vision")
    vision_module = build_modality_encoder(
        vision_encoder,
        language_model,
        modality="vision",
    )

    language_model.set_random_init()
    vision_encoder.set_random_init()
    language_model.materialize(device).to(dtype=DTYPE)
    vision_encoder.materialize(device).to(dtype=DTYPE)
    vision_module.projector.to(device=device, dtype=DTYPE)
    language_model.train()
    vision_module.train()

    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
    token_ids = configure_special_tokens(tokenizer, [IMAGE_TOKEN])
    image_token_id = token_ids[IMAGE_TOKEN]
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
        param for param in list(language_model.parameters()) + list(vision_module.parameters())
        if param.requires_grad
    )
    optimizer.zero_grad()

    total_steps = len(dataloader)
    num_warmup_steps = int(total_steps * 0.1)
    lr_scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    dataloader_iter = iter(dataloader)
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


if __name__ == "__main__":
    tyro.cli(pretrain)
