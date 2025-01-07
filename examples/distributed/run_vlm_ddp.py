from PIL import Image

import click
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.adam import Adam
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
)

from transformers.models.clip import CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPVisionModel, CLIPVisionConfig, CLIPVisionEmbeddings
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaConfig, LlamaModel

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalProjector,
    MultimodalModelProcessor,
)

def collate_fn(batch: dict[str, list], processor: MultimodalModelProcessor):
    data = processor(images=batch["images"], text=batch["texts"], return_tensors="pt", padding=True)
    data["position_ids"] = torch.arange(data["input_ids"].shape[1], device="cuda")
    data["labels"] = data["input_ids"].clone()
    data["pixel_values"] = data["pixel_values"].to(dtype=torch.bfloat16)
    data = {k: v.to("cuda") for k, v in data.items()}
    return data

class FakeDataIterator:
    def __init__(self, batch_size: int, image_size: tuple[int, int]):
        random_image = torch.randint(0, 256, image_size + (3,), dtype=torch.uint8)
        self.image = Image.fromarray(random_image.numpy())
        self.text = "text " * 256
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return {
            "images": [self.image for _ in range(self.batch_size)],
            "texts": [self.text for _ in range(self.batch_size)],
        }


@click.command
@click.option(
    "--vision_model_name_or_path",
    type=str,
    required=True,
    help="Vision model name from HF hub or local path.",
    default="openai/clip-vit-base-patch32",
)
@click.option(
    "--language_model_name_or_path",
    type=str,
    required=True,
    help="Language model name from HF hub or local path.",
    default="meta-llama/Llama-3.1-8B-Instruct",
)
def pretrain(vision_model_name_or_path: str, language_model_name_or_path: str):
    dist.init_process_group(backend="gloo")

    # Create a model
    
    if "clip" in vision_model_name_or_path:
        vision_config = CLIPVisionConfig.from_pretrained(
            vision_model_name_or_path,
            trust_remote_code=False,
            _attn_implementation="eager",
            torch_dtype=torch.bfloat16,
        )
        vision_encoder = CLIPVisionModel(vision_config).to("cuda")
        vision_encoder = ModalEncoderModule(vision_encoder)
    else:
        raise NotImplementedError

    language_config = LlamaConfig.from_pretrained(
        language_model_name_or_path,
        trust_remote_code=False,
        _attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    language_model: PreTrainedModel = LlamaForCausalLM(language_config).to("cuda")

    model: MultimodalModel = MultimodalModel(
        encoders={"vision": vision_encoder},
        language_model=language_model,
    ).to(dtype=torch.bfloat16)
    
    model.gradient_checkpointing_enable({"use_reentrant": False})

    model.train(mode=True)
    for p in model.vision_encoder.module.parameters():
        p.requires_grad_(False)
    for p in model.vision_encoder.projector.parameters():
        p.requires_grad_(True)
    for p in model.language_model.parameters():
        p.requires_grad_(False)

    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name_or_path)
    text_processor = AutoTokenizer.from_pretrained(
        language_model_name_or_path, use_fast=True
    )
    text_processor.pad_token_id = text_processor.eos_token_id
    language_model.config.pad_token_id = text_processor.pad_token_id

    processor = MultimodalModelProcessor(
        tokenizer=text_processor,
        image_processor=image_processor,
    )

    ddp_model = DistributedDataParallel(module=model, find_unused_parameters=True)
    ddp_model.train()
    optimizer = Adam([p for p in ddp_model.parameters() if p.requires_grad], lr=1e-5, fused=True)

    iterator = FakeDataIterator(batch_size=32, image_size=(1280, 720))
    with tqdm(range(5), disable=not dist.get_rank() == 0) as pbar, torch.autograd.profiler.emit_nvtx():
        for _ in pbar:
            inputs = collate_fn(next(iterator), processor)
            with torch.cuda.nvtx.range("forward()"):
                outputs = ddp_model(**inputs)
                loss = outputs.loss

            with torch.cuda.nvtx.range("backward()"):
                loss.backward()

            if dist.get_rank() == 0:
                pbar.set_postfix({"loss": loss.item()})

            with torch.cuda.nvtx.range("optimizer.step()"):
                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    pretrain()
