from __future__ import annotations

import functools
from pathlib import Path

import torch
import tyro
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import (
    AUDIO_TOKEN,
    DEFAULT_AUDIO_SAMPLE_RATE,
    DTYPE,
    IMAGE_TOKEN,
    build_modality_encoder,
    clip_vision_sequence_length,
    configure_special_tokens,
    decoder_start_token_id,
    expand_modality_tokens,
    generate_random_image,
    generate_sine_wave,
    layer_offload_config,
    optional_torch_profiler,
    tokenize_text_batch,
    vision_config_from_pretrained,
)
from cornstarch.models import (
    CornstarchExecutionPlan,
    from_hf_config,
)


class FakeDataset(Dataset):
    def __init__(
        self,
        image_size: tuple[int, int] = (720, 480),
        audio_duration: float = 10.0,
    ):
        self.image = generate_random_image(image_size)
        self.audio = generate_sine_wave(DEFAULT_AUDIO_SAMPLE_RATE, audio_duration, 440.0)
        self.text = IMAGE_TOKEN + AUDIO_TOKEN + " text" * 256

    def __len__(self) -> int:
        return 65536

    def __getitem__(self, index: int) -> dict:
        del index
        return {"image": self.image, "audio": self.audio, "text": self.text}


def _collate_valm(
    batches: list[dict],
    image_processor,
    audio_processor,
    tokenizer,
    image_sequence_length: int,
    audio_sequence_length: int,
    audio_decoder_start_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = [batch["image"] for batch in batches]
    audios = [batch["audio"] for batch in batches]
    texts = [
        expand_modality_tokens(
            batch["text"],
            {
                IMAGE_TOKEN: image_sequence_length,
                AUDIO_TOKEN: audio_sequence_length,
            },
        )
        for batch in batches
    ]

    vision_inputs = image_processor(images=images, return_tensors="pt")
    audio_inputs = audio_processor(
        audios,
        sampling_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
    )
    language_inputs = tokenize_text_batch(texts, tokenizer, device)

    return {
        "pixel_values": vision_inputs["pixel_values"].to(device=device, dtype=DTYPE),
        "input_features": audio_inputs["input_features"].to(device=device, dtype=DTYPE),
        "decoder_input_ids": torch.full(
            (len(batches), audio_sequence_length),
            audio_decoder_start_token_id,
            device=device,
            dtype=torch.long,
        ),
        **language_inputs,
    }


def _training_step(
    language_model,
    vision_module,
    audio_module,
    batch: dict[str, torch.Tensor],
    image_token_id: int,
    audio_token_id: int,
):
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=vision_module,
        pixel_values=batch["pixel_values"],
    )
    audio_outputs = plan.run_modality_encoder(
        module=audio_module,
        input_features=batch["input_features"],
        decoder_input_ids=batch["decoder_input_ids"],
        use_cache=False,
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=batch["input_ids"],
        labels=batch["labels"],
        modality_token_ids={"vision": image_token_id, "audio": audio_token_id},
        encoder_outputs={"vision": vision_outputs, "audio": audio_outputs},
    )
    language_outputs = plan.run_language_model(module=language_model, inputs=merged)
    return language_outputs.execute()


def pretrain(
    vision_encoder_name_or_path: str = "openai/clip-vit-base-patch32",
    audio_encoder_name_or_path: str = "openai/whisper-large-v3",
    llm_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct",
    use_layer_offload: bool = False,
    profile_output_path: Path | None = None,
    max_train_steps: int = 10,
):
    """Randomly initialize a vision-audio-language model with the new API."""
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print(
        "Pretraining a VALM with "
        f"{vision_encoder_name_or_path} + {audio_encoder_name_or_path} "
        f"+ {llm_name_or_path}."
    )

    vision_config = vision_config_from_pretrained(vision_encoder_name_or_path)
    audio_config = AutoConfig.from_pretrained(audio_encoder_name_or_path)
    language_config = AutoConfig.from_pretrained(llm_name_or_path)
    offload_config = layer_offload_config(use_layer_offload, device)

    language_model = from_hf_config(
        language_config,
        model_kind="language",
        layer_offload_config=offload_config,
    )
    vision_encoder = from_hf_config(
        vision_config,
        model_kind="vision",
        layer_offload_config=offload_config,
    )
    audio_encoder = from_hf_config(
        audio_config,
        model_kind="audio",
        layer_offload_config=offload_config,
    )
    vision_module = build_modality_encoder(
        vision_encoder,
        language_model,
        modality="vision",
    )
    audio_module = build_modality_encoder(
        audio_encoder,
        language_model,
        modality="audio",
    )

    language_model.set_random_init()
    vision_module.set_random_init()
    audio_module.set_random_init()
    language_model.materialize(device).to(dtype=DTYPE)
    vision_module.materialize(device).to(dtype=DTYPE)
    audio_module.materialize(device).to(dtype=DTYPE)
    language_model.train()
    vision_module.train()
    audio_module.train()

    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_name_or_path)
    audio_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
    token_ids = configure_special_tokens(tokenizer, [IMAGE_TOKEN, AUDIO_TOKEN])
    image_token_id = token_ids[IMAGE_TOKEN]
    audio_token_id = token_ids[AUDIO_TOKEN]
    image_sequence_length = clip_vision_sequence_length(vision_encoder.config)
    audio_sequence_length = 1
    audio_decoder_start_token_id = decoder_start_token_id(audio_encoder.config)

    dataset = FakeDataset(image_size=(720, 480), audio_duration=10.0)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        collate_fn=functools.partial(
            _collate_valm,
            image_processor=image_processor,
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            image_sequence_length=image_sequence_length,
            audio_sequence_length=audio_sequence_length,
            audio_decoder_start_token_id=audio_decoder_start_token_id,
            device=device,
        ),
    )
    optimizer = Adam(
        param
        for param in (
            list(language_model.parameters())
            + list(vision_module.parameters())
            + list(audio_module.parameters())
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
                    audio_module=audio_module,
                    batch=batch,
                    image_token_id=image_token_id,
                    audio_token_id=audio_token_id,
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
