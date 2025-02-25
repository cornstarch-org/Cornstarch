from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    assert (
        attention_mask is not None and attention_mask.dtype == torch.int64
    ), "Bitfield attention requires an attention mask of type torch.int64."

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # BAM follows FA2 that uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    if BitfieldUtils.sequence_lengths_cache is None:
        seq_lens = offsets = local_seq_lens = None
    else:
        seq_lens, local_seq_lens, offsets = BitfieldUtils.sequence_lengths_cache
        local_seq_lens = torch.tensor(
            local_seq_lens, dtype=torch.int64, device=query.device
        )
        seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=query.device)
        offsets = torch.nested.nested_tensor(
            [
                torch.tensor(offset, dtype=torch.int32, device=query.device)
                for offset in offsets
            ],
            device=query.device,
        ).to_padded_tensor(padding=-1)

    attn_output = bitfield_attn_func(
        query,
        key,
        value,
        None,
        None,
        attention_mask,
        local_seq_lens,
        seq_lens,
        offsets,
    )

    return attn_output, None


class BitfieldUtils:
    """
    Sequence length cache for bitfield attention that includes
    - list[int]: original sequence lengths in a batch. If sequences with a different length
        are padded to the same length, this list will contain the original lengths.
    - list[int]: local sequence lengths in a batch. If context parallelism is used,
        this will contain the partitioned sequence lengths.
        None if context parallelism is not used.
    - list[np.ndarray]: offsets for context parallelism. Each element is a numpy array
        that includes every offsets of tokens in the corresponding sequence in a batch.
        For example, [[0, 1, 2, 3, -1], [4, 5, 6, 7, 8]] indicates that there are two sequences
        in a batch, where the first one has 4 tokens and the second one has 5 tokens.
        None if context parallelism is not used.
        If sequences have different lengths, the offsets will be padded with -1.
    """

    sequence_lengths_cache: Optional[tuple[list[int], list[int], list[np.ndarray]]] = (
        None
    )

    @classmethod
    def clear_cache(cls: BitfieldUtils):
        cls.sequence_lengths_cache = None

    @classmethod
    def set_sequence_lengths_cache(
        cls: BitfieldUtils,
        sequence_lengths: list[int],
        offsets: Optional[list[np.ndarray]] = None,
        local_sequence_lengths: list[int] = None,
        overwrite: bool = True,
    ):
        if not overwrite and cls.sequence_lengths_cache is not None:
            return

        cls.sequence_lengths_cache = (sequence_lengths, local_sequence_lengths, offsets)
