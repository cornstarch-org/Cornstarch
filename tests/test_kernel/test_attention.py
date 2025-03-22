import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa

from cornstarch.kernel.attention import (
    flash_attn_func as my_flash_attn_func_triton,
)


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
    mask = torch.tril(
        torch.ones(batch_size, seqlen_q, seqlen_k, device=device, dtype=torch.bool),
        diagonal=0,
    )

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
    # mask = torch.ones(batch_size, seqlen_q, seqlen_k, device=device, dtype=torch.int64)

    reference_out = sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        mask.unsqueeze(1),
    ).transpose(1, 2)
    triton_out = my_flash_attn_func_triton(q, k, v, mask=mask)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq_triton, dk_triton, dv_triton = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dq_ref, dq_triton, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk_triton, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv_triton, rtol=5e-3, atol=5e-3)
