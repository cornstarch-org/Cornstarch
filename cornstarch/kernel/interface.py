from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.integrations.flash_attention import flash_attention_forward

from cornstarch.kernel.attention import flash_attn_func
from cornstarch.kernel.bitfield_attention import bitfield_attn_func


def create_bitfield_attention_mask(
    input_ids: torch.Tensor, token_ids: dict[str, int] = {}
) -> torch.Tensor:
    causal_bit = 1 << 62
    modal_bits = {
        modal_key: (1 << (i + 1)) for i, modal_key in enumerate(token_ids.keys())
    }
    text_bit = causal_bit | 1
    for modal_bit in modal_bits:
        text_bit |= modal_bits[modal_bit]

    attention_mask = torch.full_like(
        input_ids, text_bit, dtype=torch.int64, device=input_ids.device
    )
    for modal_key, token_id in token_ids.items():
        attention_mask[input_ids == token_id] = modal_bits[modal_key]

    return attention_mask


def cornstarch_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = flash_attn_func(
        query,
        key,
        value,
        mask=attention_mask,
    )

    return attn_output, None


def bitfield_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    # This is to detect if the model is in generation phase during inference.
    # There is no need of using bitfield attention in generation phase,
    # thus use flash attention
    if not module.training and query.shape[2] == 1:
        return flash_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask=None,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            **kwargs,
        )

    assert (
        attention_mask is not None and attention_mask.dtype == torch.int64
    ), "Bitfield attention requires an attention mask of type torch.int64."

    # BAM follows FA2 that uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    attn_output = bitfield_attn_func(query, key, value, bitfield_mask=attention_mask)

    return attn_output, None
