from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerBase

from cornstarch.models import RepeatedLayerOffloadConfig


IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
DEFAULT_AUDIO_SAMPLE_RATE = 16000
DTYPE = torch.bfloat16


def generate_random_image(image_size: tuple[int, int]) -> Image.Image:
    height, width = image_size
    image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(image)


def generate_sine_wave(
    sample_rate: int,
    duration: float,
    frequency: float = 440.0,
) -> np.ndarray:
    num_samples = int(sample_rate * duration)
    time_steps = np.linspace(0, duration, num_samples, endpoint=False)
    audio_signal = np.sin(2 * np.pi * frequency * time_steps)
    return audio_signal.astype(np.float32)


def decoder_start_token_id(audio_config: PretrainedConfig) -> int:
    return int(
        getattr(audio_config, "decoder_start_token_id", None)
        or getattr(audio_config, "bos_token_id", None)
        or 0
    )


def vision_config_from_pretrained(model_name_or_path: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model_name_or_path)
    return getattr(config, "vision_config", config)


def clip_vision_sequence_length(vision_config: PretrainedConfig) -> int:
    return (int(vision_config.image_size) // int(vision_config.patch_size)) ** 2 + 1


def expand_modality_tokens(text: str, token_counts: dict[str, int]) -> str:
    for token, count in token_counts.items():
        text = text.replace(token, " ".join([token] * count))
    return text


def configure_special_tokens(
    language_config: PretrainedConfig,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[str],
) -> dict[str, int]:
    """Add modality tokens before model construction and resize its config.

    Cornstarch models are built directly from a config, so there is no later
    ``resize_token_embeddings`` call on an HF root model. The tokenizer and
    config must therefore agree before ``from_hf_config`` creates meta
    embeddings. Otherwise newly added ``<image>``/``<audio>`` IDs can index past
    the language model's original vocabulary.
    """
    if tokenizer.eos_token is None and tokenizer.pad_token is None:
        raise ValueError("The tokenizer needs an EOS or padding token.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    language_config.vocab_size = len(tokenizer)
    token_ids = {
        token: int(tokenizer.convert_tokens_to_ids(token)) for token in tokens
    }
    if any(
        token_id < 0 or token_id >= language_config.vocab_size
        for token_id in token_ids.values()
    ):
        raise ValueError("A modality token was not added to the language vocabulary.")
    return token_ids


def tokenize_text_batch(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    language_inputs = tokenizer(texts, padding=True, return_tensors="pt")
    labels = language_inputs["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)

    return {
        "input_ids": language_inputs["input_ids"].to(device=device),
        "labels": labels.to(device=device),
    }


def layer_offload_config(
    use_layer_offload: bool,
    execution_device: torch.device,
) -> RepeatedLayerOffloadConfig | None:
    if not use_layer_offload:
        return None
    return RepeatedLayerOffloadConfig(
        enabled=True,
        execution_device=execution_device,
    )


@contextmanager
def optional_torch_profiler(profile_output_path: Path | None) -> Iterator[object | None]:
    if profile_output_path is None:
        yield None
        return

    profile_output_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True
    ) as profiler:
        yield profiler
    profiler.export_chrome_trace(str(profile_output_path))
