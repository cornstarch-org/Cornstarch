from pathlib import Path
from typing import Type

import torch
from PIL import Image
from transformers import PreTrainedModel
from transformers.models.clip import CLIPVisionModel
from transformers.models.pixtral import PixtralVisionModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)
from transformers.models.siglip import SiglipVisionModel

from cornstarch.models.multimodal_language_model import MultimodalProcessor

from transformers.image_processing_utils import BatchFeature
import logging

logger = logging.getLogger(__name__)

# from qwen_vl_utils import process_vision_info
    
def collate_fn(batches: list[dict], processor: MultimodalProcessor):
    images = []
    texts = []

    for batch in batches:
        images.append(batch["image"])
        texts.append(batch["text"])

    inputs = processor(
        encoder_inputs={"vision": {"images": images}},
        llm_inputs={"text": texts, "padding": True},
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def collate_fn_qwen25vl(batches: list[dict], processor: MultimodalProcessor):
    images = []
    texts = []

    for batch in batches:
        images.append(batch["image"])
        texts.append(batch["text"])

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=images,
        video=None, # None
        return_tensors="pt",
        padding=True,
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


# def collate_fn_scienceqa_pretrain(
#     batches: list[dict], processor: MultimodalProcessor, dataset_dir: Path
# ):
#     # NOTE: This collate function is fitted for ScienceQA dataset
#     images = []
#     texts = []

#     for batch in batches:
#         assert set(["image", "id", "conversations"]) == set(batch.keys())
#         assert isinstance(batch["conversations"], list) and len(batch["conversations"])
#         for conversation in batch["conversations"]:
#             assert ["from", "value"] == list(conversation.keys())

#         if "<image>" in batch["conversations"][0]["value"]:
#             # if file ({dataset_dir}/train/{batch['image']}) exist, open it, else open file ({dataset_dir}/val/{batch['image']})
#             try:
#                 image = Image.open(f"{dataset_dir}/train/{batch['image']}")
#             except FileNotFoundError:
#                 image = Image.open(f"{dataset_dir}/val/{batch['image']}")

#             if image.mode != "RGB":
#                 image = image.convert(mode="RGB")
#             images.append(image)

#         texts.append(
#             batch["conversations"][0]["value"]
#             + "\n"
#             + batch["conversations"][1]["value"]
#         )

#     inputs = processor(
#         encoder_inputs={"vision": {"images": images}} if images else None,
#         llm_inputs={"text": texts, "padding": True},
#         return_tensors="pt",
#     ).to(dtype=torch.bfloat16)

#     for k, v in inputs.items():
#         if isinstance(v, torch.Tensor):
#             inputs[k] = v.to("cuda").requires_grad_(v.is_floating_point())

#     inputs["labels"] = inputs["input_ids"].clone()
#     return inputs

def format_scienceqa_prompt(item: dict) -> str:
    """
    Constructs a text prompt from a ScienceQA data item.
    Adjust this based on the exact structure and desired prompt format.
    """
    question = item.get("question", "")
    choices = item.get("choices", [])
    context = item.get("hint", "") # Or 'lecture', 'solution', etc.

    prompt = f"Question: {question}\n"
    if context and context.strip():
        prompt += f"Context: {context}\n"

    if choices:
        prompt += "Options:\n"
        for i, choice in enumerate(choices):
            prompt += f"({chr(ord('A') + i)}) {choice}\n"

    prompt += "Answer:"
    return prompt

def collate_fn_scienceqa(batches: list[dict], processor) -> BatchFeature:
    """
    Collate function for ScienceQA dataset with Qwen2.5_VL.

    Args:
        batches: A list of dictionaries, where each dict represents
                 a ScienceQA sample (e.g., {'image': PIL.Image|None,
                 'question': str, 'choices': list[str], 'answer': str}).
        processor: The Qwen2VLProcessor instance.

    Returns:
        A BatchFeature dictionary ready for model input.
    """
    all_images = []
    all_messages = []
    image_present_flags = []

    for batch in batches:
        image = batch.get("image")
        question = batch.get("question")
        choices = batch.get("choices")
        answer = batch.get("answer") # The ground truth text answer
        hint = batch.get("hint")

        if not question or not answer:
            logger.warning("Skipping batch due to missing question or answer.")
            continue

        # 1. Format the text prompt
        prompt_text = format_scienceqa_prompt(batch)

        # 2. Build the messages list
        user_content = []
        has_image = False
        if image and isinstance(image, Image.Image):
            # Qwen-VL expects the image placeholder *first*
            user_content.append({"type": "image"})
            all_images.append(image)
            has_image = True
        else:
            all_images.append(None)

        user_content.append({"type": "text", "text": prompt_text})
        image_present_flags.append(has_image)

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        all_messages.append(messages)

    # 3. Apply chat template and tokenize
    # We apply the template to each message list individually
    try:
        texts_to_tokenize = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in all_messages
        ]
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        logger.error(f"Problematic messages: {all_messages}")
        raise

    # 4. Process with the Qwen2VLProcessor
    # Only pass images if any exist in the batch.
    # The processor should handle matching images to <|image_pad|> tokens
    # based on their order, but it might require all texts to expect an
    # image if *any* image is passed. Check Qwen's documentation/behavior.
    # A safer approach might be to batch image and non-image samples separately.
    # However, attempting a mixed batch:
    
    valid_images = [img for img, flag in zip(all_images, image_present_flags) if flag]
   
    inputs = processor(
        text=texts_to_tokenize,
        images=valid_images if valid_images else None, # Pass only valid images
        video=None,
        return_tensors="pt",
        padding=True,
    )

    # 5. Create labels
    # We clone input_ids as a base for labels.
    labels = inputs["input_ids"].clone()

    # --- Crucial Step: Masking Labels ---
    # We only want the model to learn to predict the 'assistant' part.
    # We need to set the 'user' part and padding tokens to -100.
    # `apply_chat_template` usually inserts tokens to delineate roles.
    # We need to find where the assistant's response *starts* for each item.
    # This often requires knowledge of the specific tokens used (e.g., <|im_start|> assistant).
    # This part can be complex and depends heavily on the processor's output.


    # Get the token ID for the start of an assistant message (example!)
    # MUST verify this token ID with your specific processor/tokenizer
    try:
        # Attempt to find a likely start token (highly dependent on tokenizer version)
        assistant_start_token_str = "<|im_start|>assistant" 
        assistant_start_token_ids = processor.tokenizer.encode(assistant_start_token_str, add_special_tokens=False) # [151644, 77091]
        
        if not assistant_start_token_ids:
             assistant_start_token_str = "assistant\n" # Fallback/Alternative
             assistant_start_token_ids = processor.tokenizer.encode(assistant_start_token_str, add_special_tokens=False) # [77091]

        if assistant_start_token_ids:
            assistant_start_id = assistant_start_token_ids[0] # [151644]
            
            for i in range(labels.shape[0]):
                input_ids_list = labels[i].tolist()
                try:
                    # Find the last occurrence, as template adds it before the answer
                    start_idx = len(input_ids_list) - 1 - input_ids_list[::-1].index(assistant_start_id)
                    # Mask everything *before* and *including* the assistant start prompt
                    labels[i, :start_idx + len(assistant_start_token_ids)] = -100
                except ValueError:
                    # If token not found, maybe mask the whole sequence or log a warning
                    logger.warning(f"Assistant start token not found in sample {i}. Labels might be incorrect.")
                    labels[i, :] = -100 # Mask all as a safety measure
        else:
            logger.error("Could not determine assistant start token ID. Labels will be unmasked.")
            
    except Exception as e:
        logger.error(f"Error during label masking: {e}. Labels will be unmasked.")

    # Also mask padding tokens: 151643
    labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels

    return inputs.to(dtype=torch.bfloat16, device="cuda")


def collate_fn_scienceqa_eval_single(batches: list[dict], processor):
    """
    Collate function for ScienceQA evaluation (batch_size=1).
    Processes a single sample, preparing it for model.generate
    and returning model inputs, ground truth, and choices.

    Args:
        batches: A list containing exactly ONE ScienceQA sample dictionary.
        processor: The Qwen2VLProcessor instance.

    Returns:
        A dictionary containing 'model_inputs', 'ground_truths', 'choices'.
    """
    # --- Ensure we are processing one by one ---
    if len(batches) != 1:
        raise ValueError(f"This collate_fn is designed for batch_size=1, but received {len(batches)} items.")

    sample = batches[0]

    # --- Extract data from the single sample ---
    image = sample.get("image")
    choices = sample.get("choices", [])
    answer_idx = sample.get("answer")
    question = sample.get("question")

    # --- Basic Validation ---
    if not question or answer_idx is None or not choices:
        logger.warning(f"Skipping sample due to missing data: {sample.get('id', 'Unknown ID')}")
        # Returning None would require filtering in the loop.
        # It's better to ensure your dataset is pre-filtered.
        # For now, we raise an error or return a special marker if this happens often.
        # Let's try to proceed but with an invalid GT if needed.
        ground_truth_text = "<<INVALID_GT>>"
    else:
        # if answer_idx is a string, use it as the ground truth
        if isinstance(answer_idx, str):
            ground_truth_text = answer_idx
        elif isinstance(answer_idx, int):
            if 0 <= answer_idx < len(choices):
                ground_truth_text = choices[answer_idx]
            else:
                logger.warning(f"Invalid answer index for sample: {sample.get('id', 'Unknown ID')}")
                ground_truth_text = "<<INVALID_GT>>"

    # --- Format the Prompt ---
    prompt_text = format_scienceqa_prompt(sample)

    # --- Build Messages for Qwen ---
    user_content = []
    # Ensure image is a PIL Image object
    if image and isinstance(image, Image.Image):
        user_content.append({"type": "image"})
    else:
        image = None # Ensure image is None if not valid PIL

    user_content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": user_content}]

    # --- Apply Chat Template for Generation ---
    text_to_tokenize = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # Crucial for generation!
    )

    # --- Process with Qwen Processor ---
    # Since it's a single item, padding isn't an issue.
    # We pass image as a list containing one item, or None.
    model_inputs = processor(
        text=text_to_tokenize,
        images=[image] if image else None,
        return_tensors="pt",
        padding=True, # Still good practice, though it won't pad much
    )

    # --- Prepare the output dictionary ---
    # We keep ground_truths and choices as lists, even though they
    # contain one item, to match the structure expected by the
    # `evaluate_scienceqa_accuracy` function's loop.
    batch_output = {
        'model_inputs': model_inputs,
        'ground_truths': [ground_truth_text],
        'choices': [choices]
    }

    return batch_output


def collate_fn_llava_pretrain(
    batches: list[dict], processor: MultimodalProcessor, dataset_dir: Path
):
    """
    Example of pretrain data sample (LCS 558K)
    {
    "id": "004539375",
    "image": "00453/004539375.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "Render a clear and concise summary of the photo.\n<image>"
      },
      {
        "from": "gpt",
        "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
      }
    ]
    },

    """
    images = []
    texts = []

    for batch in batches:
        for conversation in batch["conversations"]:
            assert ["from", "value"] == list(conversation.keys())

        if "image" in batch:
            image = Image.open(f"{dataset_dir}/{batch['image']}")
            image = image.convert("RGB")
            images.append(image)

        text = ""
        for conversation in batch["conversations"]:
            text += f"\"{conversation['from']}\"\n{conversation['value']}\n"

        texts.append(text)

    inputs = processor(
        encoder_inputs={"vision": {"images": images}} if images else None,
        llm_inputs={"text": texts, "padding": True},
        return_tensors="pt",
    ).to(dtype=torch.bfloat16, device="cuda")

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


model_names: dict[str, str] = {
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
    "pixtral": "mistral-community/pixtral-12b",
    "qwen2_vision": "Qwen/Qwen2-VL-2B-Instruct",
    "gemma2": "google/gemma-2-2b-it",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2": "Qwen/Qwen2.5-3B-Instruct",
}

vision_encoder_classes: dict[str, Type[PreTrainedModel]] = {
    "clip": CLIPVisionModel,
    "siglip": SiglipVisionModel,
    "pixtral": PixtralVisionModel,
    "qwen2_vision": Qwen2VisionTransformerPretrainedModel,
}

def collate_fn_scienceqa_llavaovhf(batches: list[dict], processor) -> BatchFeature:
    """
    Collate function for ScienceQA dataset with LLaVA-OV (Qwen2 Base).

    This function is adapted for LLaVA models that use a Qwen2-style chat template.
    The core logic remains similar to the Qwen2.5-VL collate function.

    Args:
        batches: A list of dictionaries from the ScienceQA dataset.
        processor: The AutoProcessor instance for the LLaVA model.

    Returns:
        A BatchFeature dictionary ready for model input.
    """
    all_images = []
    all_messages = []
    image_present_flags = []

    for batch in batches:
        image = batch.get("image")
        question = batch.get("question")
        choices = batch.get("choices")
        answer = batch.get("answer") # The ground truth text answer
        hint = batch.get("hint")

        if not question or not answer:
            logger.warning("Skipping batch due to missing question or answer.")
            continue

        prompt_text = format_scienceqa_prompt(batch)

        user_content = []
        has_image = False
        if image and isinstance(image, Image.Image):
            user_content.append({"type": "image"})
            all_images.append(image)
            has_image = True
        else:
            all_images.append(None)

        user_content.append({"type": "text", "text": prompt_text})
        image_present_flags.append(has_image)

        # This message structure is specific to models using the Qwen2 chat template.
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        all_messages.append(messages)

    try:
        texts_to_tokenize = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            for msgs in all_messages
        ]
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        logger.error(f"Problematic messages: {all_messages}")
        raise

    valid_images = [img for img, flag in zip(all_images, image_present_flags) if flag]
   
    inputs = processor(
        text=texts_to_tokenize,
        images=valid_images if valid_images else None,
        video=None,
        return_tensors="pt",
        padding=True,
    )

    labels = inputs["input_ids"].clone()

    # --- Label Masking for LLaVA (Qwen2 Template) ---
    # This logic assumes the model uses Qwen2's chat template tokens.
    # It finds where the assistant's response begins and masks everything before it.
    try:
        # The token sequence that precedes the assistant's response.
        assistant_start_token_str = "<|im_start|>assistant"
        assistant_start_token_ids = processor.tokenizer.encode(assistant_start_token_str, add_special_tokens=False)

        if not assistant_start_token_ids:
                assistant_start_token_str = "assistant\n" # Fallback for different tokenizer versions
                assistant_start_token_ids = processor.tokenizer.encode(assistant_start_token_str, add_special_tokens=False)

        if assistant_start_token_ids:
            # We search for the first token of the sequence from the end of the list.
            # This is a robust way to find the last occurrence, which is always the assistant's turn.
            assistant_start_id = assistant_start_token_ids[0]
            
            for i in range(labels.shape[0]):
                input_ids_list = labels[i].tolist()
                try:
                    # Find the last occurrence of the assistant start token.
                    start_idx = len(input_ids_list) - 1 - input_ids_list[::-1].index(assistant_start_id)
                    # Mask everything *before* and *including* the full assistant prompt sequence.
                    labels[i, :start_idx + len(assistant_start_token_ids)] = -100
                except ValueError:
                    logger.warning(f"Assistant start token not found in sample {i}. Labels might be incorrect.")
                    labels[i, :] = -100 # Mask all as a safety measure
        else:
            logger.error("Could not determine assistant start token ID. Labels will be unmasked.")
            
    except Exception as e:
        logger.error(f"Error during label masking: {e}. Labels will be unmasked.")

    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    return inputs.to(dtype=torch.bfloat16, device="cuda")


def collate_fn_scienceqa_eval_single_llavaovhf(batches: list[dict], processor):
    """
    Collate function for ScienceQA evaluation with LLaVA-OV (batch_size=1).
    Adapted for LLaVA models using a Qwen2-style chat template.

    Args:
        batches: A list containing exactly ONE ScienceQA sample dictionary.
        processor: The AutoProcessor for the LLaVA model.

    Returns:
        A dictionary containing 'model_inputs', 'ground_truths', 'choices'.
    """
    if len(batches) != 1:
        raise ValueError(f"This collate_fn is designed for batch_size=1, but received {len(batches)} items.")

    sample = batches[0]
    image = sample.get("image")
    choices = sample.get("choices", [])
    answer_idx = sample.get("answer")
    question = sample.get("question")

    if not question or answer_idx is None or not choices:
        logger.warning(f"Skipping sample due to missing data: {sample.get('id', 'Unknown ID')}")
        ground_truth_text = "<<INVALID_GT>>"
    else:
        if isinstance(answer_idx, str):
            ground_truth_text = answer_idx
        elif isinstance(answer_idx, int):
            if 0 <= answer_idx < len(choices):
                ground_truth_text = choices[answer_idx]
            else:
                logger.warning(f"Invalid answer index for sample: {sample.get('id', 'Unknown ID')}")
                ground_truth_text = "<<INVALID_GT>>"

    prompt_text = format_scienceqa_prompt(sample)

    # --- Build Messages for LLaVA (Qwen2 Template) ---
    user_content = []
    if image and isinstance(image, Image.Image):
        user_content.append({"type": "image"})
    else:
        image = None

    user_content.append({"type": "text", "text": prompt_text})
    
    # This message structure is specific to the Qwen2 chat template.
    messages = [{"role": "user", "content": user_content}]

    # --- Apply Chat Template for Generation ---
    text_to_tokenize = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # Crucial for generation!
    )

    model_inputs = processor(
        text=text_to_tokenize,
        images=[image] if image else None,
        return_tensors="pt",
        padding=True,
    )

    batch_output = {
        'model_inputs': model_inputs,
        'ground_truths': [ground_truth_text],
        'choices': [choices]
    }

    return batch_output
