import functools

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.siglip.modeling_siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
)
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProcessor,
)


def generate_sine_wave(
    sample_rate: int, duration: float, frequency: float = 440.0
) -> np.ndarray:
    """
    Parameters:
        sample_rate (int): Sampling rate in Hz.
        duration (float): Duration of the sine wave in seconds.
        frequency (float): Frequency of the sine wave in Hz. Default is 440.0 Hz.

    Returns:
        np.ndarray: NumPy array containing the audio signal (float values between -1.0 and 1.0).
    """
    # Calculate the total number of samples
    num_samples = int(sample_rate * duration)
    # Create a time array from 0 to duration (excluded) with num_samples points
    t = np.linspace(0, duration, num_samples, endpoint=False)
    # Generate sine wave values for each time point
    audio_signal = np.sin(2 * np.pi * frequency * t)
    return audio_signal.astype(np.float32)  # use 32-bit float for consistency


def generate_random_image(resolution: tuple[int, int]) -> np.ndarray:
    """
    Generate a random RGB image.

    Parameters:
        resolution (tuple): A tuple of two integers (width, height).

    Returns:
        np.ndarray: A random image with shape (height, width, 3) and dtype uint8.
    """
    # Unpack resolution assuming it's given as (width, height)
    width, height = resolution
    # Create an array with random integers in [0, 255] for 3 color channels (RGB)
    image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
    return image


class FakeDataset(Dataset):
    def __init__(
        self, image_size: tuple[int, int] = (720, 480), audio_duration: float = 10.0
    ):
        self.image = generate_random_image(image_size)
        self.audio = generate_sine_wave(16000, audio_duration, 440.0)

        # later tokens will be appended after processing it
        self.text = "<image><audio>" + " text" * 256

    def __len__(self):
        return 65536

    def __getitem__(self, index: int) -> dict:
        return {"image": self.image, "audio": self.audio, "text": self.text}


def collate_fn(batches: list[dict], processor: MultimodalProcessor, seqlen: int = 1024):
    images = []
    texts = []
    audios = []

    token = processor.llm_tokenizer.pad_token
    for batch in batches:
        images.append(batch["image"])
        texts.append(token * 256 + "<image>" + token * 256 + "<audio>" + token * 512)
        audios.append(batch["audio"])

    inputs = processor(
        encoder_inputs={
            "vision": {"images": images},
            "audio": {"raw_speech": audios, "sampling_rate": 16000},
        },
        llm_inputs={"text": texts, "padding": True},
        return_tensors="pt",
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    for value in inputs.values():
        value.requires_grad_(value.is_floating_point())

    return inputs.data


def pretrain():
    """
    Training example for Siglip and Whisper
    """
    torch.cuda.set_device(0)

    vision_encoder_path = "google/siglip-so400m-patch14-384"
    audio_encoder_path = "openai/whisper-large-v3"
    llm_name_or_path = "meta-llama/Llama-3.2-3B-Instruct"

    with torch.device("meta"):
        vision_config = SiglipVisionConfig.from_pretrained(vision_encoder_path)
        vision_encoder = SiglipVisionModel(vision_config)

        audio_config = WhisperConfig.from_pretrained(audio_encoder_path)
        audio_encoder = WhisperEncoder(audio_config)

        llm_config = AutoConfig.from_pretrained(llm_name_or_path)
        language_model = AutoModelForCausalLM.from_config(llm_config)

        model = MultimodalModel(
            encoders={
                "vision": ModalEncoderModule(vision_encoder),
                "audio": ModalEncoderModule(audio_encoder),
            },
            language_model=language_model,
        ).to(dtype=torch.bfloat16)

    model.gradient_checkpointing_enable()
    model.train()

    model.to_empty(device="cuda")

    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_path)
    audio_processor = AutoFeatureExtractor.from_pretrained(audio_encoder_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    processor = MultimodalProcessor(
        encoder_processors={
            "vision": image_processor,
            "audio": audio_processor,
        },
        llm_tokenizer=tokenizer,
        model=model,
        predefined_tokens={
            "vision": "<image>",
            "audio": "<audio>",
        },
    )

    dataset = FakeDataset(image_size=(720, 480), audio_duration=10.0)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        collate_fn=functools.partial(collate_fn, processor=processor),
    )

    total_steps = len(dataloader)
    dataloader_iter = iter(dataloader)

    with tqdm(range(total_steps)) as pbar:
        for item in pbar:
            inputs = next(dataloader_iter)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()


if __name__ == "__main__":
    pretrain()
