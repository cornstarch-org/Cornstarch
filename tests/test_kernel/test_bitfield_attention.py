import pytest
import torch
import triton
import triton.language as tl
from torch.nn.functional import scaled_dot_product_attention as sdpa

from cornstarch.kernel.bitfield_attention import (
    _materialize_bitfield_mask_block,
    bitfield_attn_func,
)

BLOCK_M: tl.constexpr = 128
BLOCK_N: tl.constexpr = 32


@triton.jit
def materialize_bitfield_mask_block(
    Bitfield_mask: tl.tensor,
    Out: tl.tensor,
    stride_maskb,
    stride_outb,
    stride_outm,
    seqlen,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    start_m = tl.program_id(1)
    start_n = tl.program_id(2)

    submask = _materialize_bitfield_mask_block(
        Bitfield_mask + off_b * stride_maskb,
        start_m * BLOCK_M,
        start_n * BLOCK_N,
        seqlen,
        seqlen,
        None,
        None,
        BLOCK_M,
        BLOCK_N,
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptr = (
        Out + off_b * stride_outb + offs_m[:, None] * stride_outm + offs_n[None, :]
    )
    tl.store(
        out_ptr,
        submask,
        mask=(offs_m < seqlen)[:, None] & (offs_n < seqlen)[None, :],
    )


@pytest.mark.parametrize("size", ["small", "large"])
def test_materialize_bitfield_mask(size: str):
    bitfield_mask: torch.Tensor
    expected_full_mask: torch.Tensor
    device = torch.device("cuda")
    batch_size = 2

    seq_len = 64 if size == "small" else 256
    bitfield_mask = torch.full(
        (batch_size, seq_len),
        (1 << 62) | 1 | (1 << 1) | (1 << 2),
        device=device,
        dtype=torch.int64,
    )
    expected_full_mask = torch.tril(
        torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=0,
    )

    if size == "small":
        bitfield_mask[0, 12:24] = 1 << 1
        bitfield_mask[1, 4:16] = 1 << 1
        bitfield_mask[1, 36:56] = 1 << 2
        expected_full_mask[0, 12:24, :] = False
        expected_full_mask[0, 12:24, 12:24] = True
        expected_full_mask[1, 4:16, :] = False
        expected_full_mask[1, 4:16, 4:16] = True
        expected_full_mask[1, 36:56, :] = False
        expected_full_mask[1, 36:56, 36:56] = True
    else:
        bitfield_mask[0, 50:150] = 1 << 1
        bitfield_mask[1, 48:120] = 1 << 1
        bitfield_mask[1, 164:200] = 1 << 2
        expected_full_mask[0, 50:150, :] = False
        expected_full_mask[0, 50:150, 50:150] = True
        expected_full_mask[1, 48:120, :] = False
        expected_full_mask[1, 48:120, 48:120] = True
        expected_full_mask[1, 164:200, :] = False
        expected_full_mask[1, 164:200, 164:200] = True

    converted_full_mask = torch.empty(
        (batch_size, seq_len, seq_len), dtype=torch.bool, device=device
    )
    grid = (batch_size, triton.cdiv(seq_len, BLOCK_M), triton.cdiv(seq_len, BLOCK_N))
    materialize_bitfield_mask_block[grid](
        bitfield_mask,
        converted_full_mask,
        bitfield_mask.stride(0),
        converted_full_mask.stride(0),
        converted_full_mask.stride(1),
        seq_len,
        BLOCK_M,
        BLOCK_N,
    )

    assert (converted_full_mask == expected_full_mask).all()


@pytest.mark.parametrize("head_dim", [32, 64, 128], ids=lambda x: f"hd={x}")
@pytest.mark.parametrize(
    "seqlen", [57, 128, 144, 256, 283, 512, 1024], ids=lambda x: f"sl={x}"
)
@pytest.mark.parametrize("batch_size", [1, 2, 4], ids=lambda x: f"b={x}")
def test_bitfield_attention(head_dim: int, seqlen: int, batch_size: int):
    device = torch.device("cuda")
    dtype = torch.float16
    num_heads = 6

    torch.random.manual_seed(0)

    q = (
        torch.randn(
            batch_size,
            seqlen,
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
            seqlen,
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
            seqlen,
            num_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_(True)
    )

    # create a bitfield attention mask
    def get_bitfield_attention_mask() -> tuple[torch.Tensor, torch.Tensor]:
        text_bit = (1 << 62) | 1 | (1 << 1) | (1 << 2)

        attention_mask = torch.full(
            (batch_size, seqlen),
            text_bit,
            dtype=torch.int64,
            device=device,
        )
        # attention_mask[:, 12:24] = 1 << 1
        # attention_mask[:, 36:56] = 1 << 2

        full_mask = torch.tril(
            torch.ones((batch_size, seqlen, seqlen), dtype=torch.bool, device=device),
            diagonal=0,
        )
        # full_mask[:, 12:24, :] = False
        # full_mask[:, 12:24, 12:24] = True
        # full_mask[:, 36:56, :] = False
        # full_mask[:, 36:56, 36:56] = True

        return attention_mask, full_mask

    bitfield_mask, full_mask = get_bitfield_attention_mask()

    reference_out = sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), full_mask.unsqueeze(1)
    ).transpose(1, 2)

    # without materializing the full mask,
    # our Triton implementation accepts the bitfield mask directly
    # and internally materializes the mask in a tiled fashion
    triton_out = bitfield_attn_func(q, k, v, bitfield_mask=bitfield_mask)
    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(dq_ref, dq, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("head_dim", [32, 64, 128], ids=lambda x: f"hd={x}")
@pytest.mark.parametrize(
    "seqlen", [57, 128, 144, 256, 283, 512, 1024], ids=lambda x: f"sl={x}"
)
@pytest.mark.parametrize("batch_size", [1, 2], ids=lambda x: f"b={x}")
@pytest.mark.parametrize("num_kv_heads", [1, 2, 3], ids=lambda x: f"kvh={x}")
def test_bitfield_attention_gqa(
    head_dim: int, seqlen: int, batch_size: int, num_kv_heads: int
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_attention_heads = 6
    assert num_attention_heads % num_kv_heads == 0
    assert num_attention_heads >= num_kv_heads

    torch.random.manual_seed(0)

    q = (
        torch.randn(
            batch_size,
            seqlen,
            num_attention_heads,
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
            seqlen,
            num_kv_heads,
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
            seqlen,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        .normal_(mean=0, std=0.5)
        .requires_grad_(True)
    )

    # create a bitfield attention mask
    def get_bitfield_attention_mask() -> tuple[torch.Tensor, torch.Tensor]:
        text_bit = (1 << 62) | 1 | (1 << 1) | (1 << 2)

        attention_mask = torch.full(
            (batch_size, seqlen),
            text_bit,
            dtype=torch.int64,
            device=device,
        )
        attention_mask[:, 12:24] = 1 << 1
        attention_mask[:, 36:56] = 1 << 2

        full_mask = torch.tril(
            torch.ones((batch_size, seqlen, seqlen), dtype=torch.bool, device=device),
            diagonal=0,
        )
        full_mask[:, 12:24, :] = False
        full_mask[:, 12:24, 12:24] = True
        full_mask[:, 36:56, :] = False
        full_mask[:, 36:56, 36:56] = True

        return attention_mask, full_mask

    bitfield_mask, full_mask = get_bitfield_attention_mask()

    reference_out = sdpa(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        full_mask.unsqueeze(1),
        enable_gqa=True,
    ).transpose(1, 2)

    # without materializing the full mask,
    # our Triton implementation accepts the bitfield mask directly
    # and internally materializes the mask in a tiled fashion
    triton_out = bitfield_attn_func(q, k, v, bitfield_mask=bitfield_mask)
    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(dq_ref, dq, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv, rtol=5e-3, atol=5e-3)
