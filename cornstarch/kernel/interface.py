import torch


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
