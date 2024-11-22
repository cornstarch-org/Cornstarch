import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)


def convert_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    assert (
        attention_mask.ndim == 3
    ), f"Expected 4d attention mask of shape (batch_size, q_len, kv_len), got {attention_mask.shape}"

    def custom_mask_mod(b, h, q_idx, kv_idx):
        return attention_mask[b, q_idx, kv_idx].bool()

    block_mask = create_block_mask(
        mask_mod=custom_mask_mod,
        B=attention_mask.shape[0],
        H=num_heads,
        Q_LEN=attention_mask.shape[1],
        KV_LEN=attention_mask.shape[2],
        device=attention_mask.device,
        BLOCK_SIZE=min(attention_mask.shape[1], 128),
        _compile=True,
    )

    return block_mask


def convert_legacy_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    """
    Convert legacy attention mask to BlockMask.
    Legacy attention mask refers to an attention mask with either 0 or 1,
    and later converted to a 4D causal mask.

    Args:
        attention_mask: a 2D tensor of shape (batch_size, seq_len) and type torch.bool or torch.int8.
    """
    assert not getattr(
        attention_mask, "cornstarch_is_bitattention", False
    ), "Expected Non BitAttentionMask"
    assert (
        attention_mask.ndim == 2
    ), f"Expected 2D attention mask, got {attention_mask.ndim}"

    bsz, seq_len = attention_mask.shape

    def causalmask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        return q_idx >= kv_idx

    return create_block_mask(
        causalmask_mod,
        bsz,
        num_heads,
        seq_len,
        seq_len,
        device=attention_mask.device,
        BLOCK_SIZE=min(seq_len, 128),
        _compile=True,
    )


def convert_bit_attention_mask_to_block_mask(
    attention_mask: torch.Tensor, num_heads: int
) -> BlockMask:
    """
    Convert BitAttentionMask to BlockMask.
    BitAttentionMask refers to an attention mask, where each element is torch.int64 type
    and each bit represents a different types of token to attend.

    Args:
        attention_mask: a 2D tensor of shape (batch_size, seq_len) and type torch.int64.
    """
    assert (
        attention_mask.dtype == torch.int64
    ), f"Expected torch.int64, got {attention_mask.dtype}"
    assert getattr(
        attention_mask, "cornstarch_is_bitattention", False
    ), "Expected BitAttentionMask"
    assert (
        attention_mask.ndim == 2
    ), f"Expected 2D attention mask, got {attention_mask.ndim}"

    bsz, seq_len = attention_mask.shape

    def anymask_mod(
        b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        is_causal = (attention_mask[b][q_idx] & (1 << 62)) != 0
        # Remove it as of now. TODO: Add support for non-contiguous attention
        # attend_non_contiguous = (attention_mask[b][q_idx] & (1 << 61)) != 0

        return (is_causal & (q_idx >= kv_idx)) | (
            ~is_causal & (attention_mask[b, q_idx] == attention_mask[b, kv_idx])
        )

    return create_block_mask(
        anymask_mod,
        bsz,
        num_heads,
        seq_len,
        seq_len,
        device=attention_mask.device,
        BLOCK_SIZE=min(seq_len, 128),
        _compile=True,
    )
