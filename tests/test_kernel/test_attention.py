import math

import pytest
import torch
from einops import rearrange, repeat

from cornstarch.kernel.attention import (
    flash_attn_func as my_flash_attn_func_triton,
)


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        attention_mask: (seqlen_q, seqlen_k)

    Return:
        output: (batch_size, seqlen_q, nheads, head_dim)
    """
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "Number of heads in q must be divisible by number of heads in k and v"
        )
    num_key_value_groups = q.shape[1] // k.shape[1]

    if num_key_value_groups > 1:
        from transformers.models.llama.modeling_llama import repeat_kv

        k = repeat_kv(k, num_key_value_groups)
        v = repeat_kv(v, num_key_value_groups)

    head_dim = q.shape[-1]
    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        # attn_weights should be (batch_size, nheads, seqlen_q, seqlen_k)
        # mask out with -inf for every (batch_size, nheads)
        # using pattern of "q k -> 1 1 q k" and add it to attn_weights
        # Expand to match the dimensions of attn_weights and attention_mask
        attn_weights = attn_weights + torch.where(
            attention_mask.unsqueeze(1).expand(-1, attn_weights.shape[1], -1, -1) == 0,
            float("-inf"),
            0.0,
        )

    # upcast attention to fp32
    attn_weights = torch.nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(q.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=0.0, training=True)
    attn_output = torch.matmul(attn_weights, v)

    assert attn_output.shape == q.shape

    return attn_output.transpose(1, 2).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen",
    [
        (128, 128),
        (128, 1024),
        (1024, 128),
        (1024, 1024),
        (57, 57),
        (144, 144),
        (283, 283),
    ],
    ids=["128x128", "128x1024", "1024x128", "1024x1024", "57x57", "144x144", "283x283"],
)
def test_my_flash_attention(dtype: torch.dtype, head_dim: int, seqlen: tuple[int, int]):
    device = torch.device("cuda")
    batch_size = 1
    num_heads = 12

    seqlen_q, seqlen_k = seqlen

    torch.random.manual_seed(0)

    q = (
        torch.randn(
            batch_size,
            seqlen_q,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_(True)
    )
    k = (
        torch.randn(
            batch_size,
            seqlen_k,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_(True)
    )
    v = (
        torch.randn(
            batch_size,
            seqlen_k,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_(True)
    )

    # 1. Random mask
    # mask = torch.randint(0, 2, (batch_size, seqlen_q, seqlen_k), device=device, dtype=torch.int64)

    # 2. Causal mask
    # mask = torch.tril(
    #     torch.ones(batch_size, seqlen_q, seqlen_k, device=device, dtype=torch.int64),
    #     diagonal=0,∏
    # )

    # 3. Multimodal mask
    # mask = torch.tril(
    #     torch.ones(batch_size, seqlen_q, seqlen_k, device=device, dtype=torch.int64),
    #     diagonal=0,
    # )
    # mask[:, 12:24, :] = 0
    # mask[:, 12:24, 12:24] = 1
    # mask[:, 36:56, :] = 0
    # mask[:, 36:56, 36:56] = 1

    # 4. Full mask
    mask = torch.ones(batch_size, seqlen_q, seqlen_k, device=device, dtype=torch.int64)

    reference_out = reference_attention(q, k, v, mask)
    triton_out = my_flash_attn_func_triton(q, k, v)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq_triton, dk_triton, dv_triton = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dq_ref, dq_triton, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk_triton, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv_triton, rtol=5e-3, atol=5e-3)
