import io

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def create_random_image(width: int, height: int) -> Image:
    """
    Creates a fake JPEG image with random noise of specified width and height.

    Parameters:
    width (int): The width of the image.
    height (int): The height of the image.

    Returns:
    PIL.JpegImagePlugin.JpegImageFile: An instance of a JPEG image with random noise.
    """
    # Generate random noise for the image (values between 0 and 255 for RGB channels)
    random_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    # Create an image from the random data
    image = Image.fromarray(random_data, "RGB")

    # Save the image to a BytesIO object to simulate a JPEG file
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    # Open the image from the BytesIO object as a JpegImageFile instance
    jpeg_image = Image.open(image_bytes)

    return jpeg_image


def generate_causal_blockwise_mask(seq_len):
    """
    Generate causal blockwise mask where:
    - Each block attends only to itself within the block (causally)
    - The last block attends to all previous tokens
    """
    mask = torch.zeros((seq_len, seq_len), device="cuda")

    max_block_size = seq_len // 16
    min_block_size = max_block_size // 4

    current_pos = 0
    blocks = []  # Keep track of block positions

    while current_pos < seq_len:
        remaining = seq_len - current_pos

        if remaining <= min_block_size:
            block_size = remaining
        else:
            max_size = min(max_block_size, remaining)
            block_size = torch.randint(min_block_size, max_size + 1, (1,)).item()

        end_pos = current_pos + block_size
        blocks.append((current_pos, end_pos))

        # For each position in current block
        for i in range(current_pos, end_pos):
            # Always attend up to current position within block
            mask[i, current_pos : i + 1] = 1

        current_pos = end_pos

    # Special handling for last block: attend to all previous tokens
    last_start, last_end = blocks[-1]
    for i in range(last_start, last_end):
        mask[i, : i + 1] = 1

    return mask


def generate_prefix_lm_document_mask(seq_len):
    """
    Generate prefix LM document mask where:
    - First block: prefix LM causal (attends causally within block)
    - Middle blocks: document mask only (bidirectional within block)
    - Last block: prefix LM causal (attends to all previous + causal within block)
    """
    mask = torch.zeros((seq_len, seq_len), device="cuda")

    max_block_size = seq_len // 16
    min_block_size = max_block_size // 4

    current_pos = 0
    blocks = []  # Keep track of block positions

    while current_pos < int(seq_len * 0.9):
        remaining = seq_len - current_pos

        if remaining <= min_block_size:
            block_size = remaining
        else:
            max_size = min(max_block_size, remaining)
            block_size = torch.randint(min_block_size, max_size + 1, (1,)).item()

        end_pos = current_pos + block_size
        blocks.append((current_pos, end_pos))
        current_pos = end_pos

    # Handle first block (prefix LM causal)
    # first_start, first_end = blocks[0]
    # for i in range(first_start, first_end):
    #     mask[i, first_start:i+1] = 1  # Causal attention within first block
    mask[: blocks[0][1], : blocks[0][1]] = generate_prefix_lm_causal_mask(blocks[0][1])

    # Handle middle blocks (document mask only)
    for start, end in blocks[1:-1]:
        # Only attend within own block (bidirectional)
        mask[start:end, start:end] = 1

    # Handle last block (prefix LM causal)
    last_start, last_end = blocks[-1][0], seq_len
    # Attend to all previous tokens + causal within block
    for i in range(last_start, last_end):
        mask[i, : i + 1] = 1
    # mask[last_start:last_end, last_start:last_end] = generate_prefix_lm_causal_mask(
    #     last_end - last_start
    # )

    return mask


def generate_encoder_embedded_mask(seq_len):
    mask = torch.zeros((seq_len, seq_len), device="cuda")

    current_pos = 0
    blocks = []
    causals = []

    max_block_size = seq_len // 16
    min_block_size = max_block_size // 4

    # Handle first causal mask
    first_causal_size = torch.randint(
        int(seq_len * 0.1), int(seq_len * 0.3), (1,)
    ).item()
    causals.append((0, first_causal_size))
    current_pos = first_causal_size

    while current_pos < seq_len * 0.9:
        remaining = seq_len - current_pos

        if remaining <= min_block_size:
            block_size = remaining
        else:
            max_size = min(max_block_size, remaining)
            block_size = torch.randint(min_block_size, max_size + 1, (1,)).item()

        # Add block
        end_pos = current_pos + block_size
        blocks.append((current_pos, end_pos))
        current_pos = end_pos

        # Add causal mask between blocks
        max_size = min(seq_len - current_pos, max_block_size)
        min_size = min(min_block_size, max_size)
        causal_size = torch.randint(min_size, max_size + 1, (1,)).item()

        end_pos = current_pos + causal_size
        causals.append((current_pos, end_pos))
        current_pos = end_pos

    # Handle middle blocks
    for start, end in blocks:
        mask[start:end, start:end] = 1

    # Handle causal blocks
    for start, end in causals:
        for i in range(start, end):
            mask[i, : i + 1] = 1

    # Handle last block as causal
    # last_start, last_end = blocks[-1][0], seq_len
    # for i in range(last_start, last_end):
    #     mask[i, : i + 1] = 1

    return mask


def generate_prefix_lm_causal_mask(seq_len, prefix_len=None):
    """
    Generate prefix LM causal mask where prefix tokens can attend to each other,
    and other tokens can attend to prefix and previous tokens
    """
    mask = torch.zeros((seq_len, seq_len), device="cuda")

    # If prefix_len not specified, randomly choose between 10-80% of seq_len
    if prefix_len is None:
        min_len = seq_len // 10
        prefix_len = torch.randint(min_len, seq_len, (1,)).item()

    # Prefix tokens can attend to all prefix tokens
    mask[:prefix_len, :prefix_len] = 1

    # Other tokens follow causal attention
    for i in range(prefix_len, seq_len):
        mask[i, : i + 1] = 1

    return mask


def generate_multimodal_packed_mask(seq_len):
    """
    Generate a packed mask for multimodal fusion.
    """
    num_pack = 4
    mask = torch.zeros((seq_len, seq_len), device="cuda")

    # Calculate num_pack numbers where the sume is seq_len
    pack_sizes = torch.randint(seq_len // 10, seq_len // 4, (num_pack,))
    pack_sizes = pack_sizes / pack_sizes.sum() * seq_len
    pack_sizes = pack_sizes.int().tolist()

    mask_generators = {
        "prefix_lm_document": generate_prefix_lm_document_mask,
        "prefix_lm_causal": generate_prefix_lm_causal_mask,
        "encoder_embedded": generate_encoder_embedded_mask,
    }

    current_pos = 0
    for pack_size in pack_sizes:

        mask_generator = torch.randint(0, len(mask_generators), (1,)).item()
        mask_generator = mask_generators[list(mask_generators.keys())[mask_generator]]

        mask[
            current_pos : current_pos + pack_size, current_pos : current_pos + pack_size
        ] = mask_generator(pack_size)

        current_pos += pack_size

    return mask


def get_random_mask(seq_len, mask_type="causal_blockwise"):
    """
    Main function to generate a random mask of specified type
    """
    mask_generators = {
        "causal_blockwise": generate_causal_blockwise_mask,
        "prefix_lm_document": generate_prefix_lm_document_mask,
        "prefix_lm_causal": generate_prefix_lm_causal_mask,
        "encoder_embedded": generate_encoder_embedded_mask,
        "multimodal_packed": generate_multimodal_packed_mask,
    }

    if mask_type not in mask_generators:
        raise ValueError(f"mask_type must be one of {list(mask_generators.keys())}")

    return mask_generators[mask_type](seq_len)


def visualize_masks(seq_len=32, num_samples=10):
    """
    Visualize multiple samples of each mask type
    Args:
        seq_len: Sequence length for visualization
        num_samples: Number of different masks to generate for each type
    """
    mask_types = [
        # "causal_blockwise",
        "prefix_lm_document",
        "prefix_lm_causal",
        "encoder_embedded",
        "multimodal_packed",
    ]

    # Create subplot grid: 2 rows (mask types) x 10 columns (samples)
    fig, axes = plt.subplots(len(mask_types), num_samples, figsize=(30, 10))
    cmap = matplotlib.colors.ListedColormap(["#CBCBCB", "#7EA6E1"])

    for type_idx, mask_type in enumerate(mask_types):
        for sample_idx in range(num_samples):
            mask = get_random_mask(seq_len, mask_type)
            axes[type_idx, sample_idx].imshow(mask.numpy(), cmap=cmap)
            if sample_idx == 0:  # Only show type label on first column
                axes[type_idx, sample_idx].set_ylabel(
                    mask_type.replace("_", " ").title()
                )
            axes[type_idx, sample_idx].axis("on")
            axes[type_idx, sample_idx].set_xticks([])
            axes[type_idx, sample_idx].set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # seq_lengths = [8192, 16384, 32768, 65536]

    # only for visualization
    seq_lengths = [512, 1024]

    for seq_len in seq_lengths:
        print(f"\nVisualizing masks for sequence length: {seq_len}")
        visualize_masks(seq_len)

    # for Usage
    seq_len = 512
    get_random_mask(seq_len, "causal_blockwise")
    get_random_mask(seq_len, "prefix_lm_document")
    get_random_mask(seq_len, "prefix_lm_causal")
