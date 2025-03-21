import pytest
import torch
import triton
import triton.language as tl
from torch.nn.functional import scaled_dot_product_attention as sdpa

from cornstarch.kernel.bitfield_attention import (
    bitfield_attn_func,
    get_submask_from_bitfield_mask,
)

BLOCK_SIZE: tl.constexpr = 64


@triton.jit
def func_wrapper(
    bitfield_mask,
    output_buffer,
    q_range_start: tl.constexpr,
    q_range_end: tl.constexpr,
    kv_range_start: tl.constexpr,
    kv_range_end: tl.constexpr,
    stride_bamb: tl.constexpr,
    stride_outb: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs_m = tl.arange(q_range_start, q_range_end)
    offs_n = tl.arange(kv_range_start, kv_range_end)
    off_b = tl.program_id(1)  # blockIdx.y

    output = get_submask_from_bitfield_mask(
        tl.load(bitfield_mask + off_b * stride_bamb + offs_m),
        tl.load(bitfield_mask + off_b * stride_bamb + offs_n),
        offs_m,
        offs_n,
    )

    # We expect output buffer already targets the correct block
    output_ptrs = (
        output_buffer
        + off_b * stride_outb
        + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    tl.store(output_ptrs, output)


@pytest.mark.parametrize("size", ["small", "large"])
def test_get_submask_from_bitfield_mask_multibatch(size: str):
    bitfield_mask: torch.Tensor
    expected_submask: torch.Tensor
    device = torch.device("cuda")
    batch_size = 2

    seq_len = 64 if size == "small" else 256
    bitfield_mask = torch.full(
        (batch_size, seq_len),
        (1 << 62) | 1 | (1 << 1) | (1 << 2),
        device=device,
        dtype=torch.int64,
    )
    expected_submask = torch.tril(
        torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=0,
    )

    if size == "small":
        bitfield_mask[0, 12:24] = 1 << 1
        bitfield_mask[1, 4:16] = 1 << 1
        bitfield_mask[1, 36:56] = 1 << 2
        expected_submask[0, 12:24, :] = False
        expected_submask[0, 12:24, 12:24] = True
        expected_submask[1, 4:16, :] = False
        expected_submask[1, 4:16, 4:16] = True
        expected_submask[1, 36:56, :] = False
        expected_submask[1, 36:56, 36:56] = True
    else:
        bitfield_mask[0, 50:150] = 1 << 1
        bitfield_mask[1, 48:120] = 1 << 1
        bitfield_mask[1, 164:200] = 1 << 2
        expected_submask[0, 50:150, :] = False
        expected_submask[0, 50:150, 50:150] = True
        expected_submask[1, 48:120, :] = False
        expected_submask[1, 48:120, 48:120] = True
        expected_submask[1, 164:200, :] = False
        expected_submask[1, 164:200, 164:200] = True

    grid = lambda META: (1, batch_size)
    for i in range(0, seq_len, BLOCK_SIZE):
        for j in range(0, seq_len, BLOCK_SIZE):
            submask = torch.empty(
                (batch_size, BLOCK_SIZE, BLOCK_SIZE), dtype=torch.bool, device=device
            )

            func_wrapper[grid](
                bitfield_mask,
                submask,
                i,
                i + BLOCK_SIZE,
                j,
                j + BLOCK_SIZE,
                bitfield_mask.stride(0),
                submask.stride(0),
                BLOCK_SIZE,
            )
            torch.testing.assert_close(
                submask, expected_submask[:, i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            )


@pytest.mark.parametrize("type", ["causal", "one_modality", "two_modalities"])
@pytest.mark.parametrize("size", ["small", "large"])
def test_get_submask_from_bitfield_mask(type: str, size: str):
    bitfield_mask: torch.Tensor
    expected_submask: torch.Tensor
    device = torch.device("cuda")
    batch_size = 1

    if type == "causal":
        if size == "small":
            bitfield_mask = torch.full(
                (batch_size, 64), (1 << 62) | 1, device=device, dtype=torch.int64
            )
            expected_submask = torch.tril(
                torch.ones((64, 64), dtype=torch.bool, device=device),
                diagonal=0,
            )
        else:
            bitfield_mask = torch.full(
                (batch_size, 256), (1 << 62) | 1, device=device, dtype=torch.int64
            )
            expected_submask = torch.tril(
                torch.ones((256, 256), dtype=torch.bool, device=device),
                diagonal=0,
            )
    elif type == "one_modality":
        if size == "small":
            bitfield_mask = torch.full(
                (batch_size, 64),
                (1 << 62) | 1 | (1 << 1),
                device=device,
                dtype=torch.int64,
            )
            bitfield_mask[:, 12:24] = 1 << 1
            expected_submask = torch.tril(
                torch.ones((64, 64), dtype=torch.bool, device=device),
                diagonal=0,
            )
            expected_submask[12:24, :] = False
            expected_submask[12:24, 12:24] = True
        else:
            bitfield_mask = torch.full(
                (batch_size, 256),
                (1 << 62) | 1 | (1 << 1),
                device=device,
                dtype=torch.int64,
            )
            bitfield_mask[:, 48:150] = 1 << 1
            expected_submask = torch.tril(
                torch.ones((256, 256), dtype=torch.bool, device=device),
                diagonal=0,
            )
            expected_submask[48:150, :] = False
            expected_submask[48:150, 48:150] = True
    else:
        if size == "small":
            bitfield_mask = torch.full(
                (batch_size, 64),
                (1 << 62) | 1 | (1 << 1) | (1 << 2),
                device=device,
                dtype=torch.int64,
            )
            bitfield_mask[:, 12:24] = 1 << 1
            bitfield_mask[:, 36:56] = 1 << 2
            expected_submask = torch.tril(
                torch.ones((64, 64), dtype=torch.bool, device=device),
                diagonal=0,
            )
            expected_submask[12:24, :] = False
            expected_submask[12:24, 12:24] = True
            expected_submask[36:56, :] = False
            expected_submask[36:56, 36:56] = True
        else:
            bitfield_mask = torch.full(
                (batch_size, 256),
                (1 << 62) | 1 | (1 << 1) | (1 << 2),
                device=device,
                dtype=torch.int64,
            )
            bitfield_mask[:, 48:120] = 1 << 1
            bitfield_mask[:, 164:200] = 1 << 2
            expected_submask = torch.tril(
                torch.ones((256, 256), dtype=torch.bool, device=device),
                diagonal=0,
            )
            expected_submask[48:120, :] = False
            expected_submask[48:120, 48:120] = True
            expected_submask[164:200, :] = False
            expected_submask[164:200, 164:200] = True

    # Call the kernel "per block" manually to simulate submask creation.
    seq_len = 64 if size == "small" else 256
    grid = lambda META: (1, batch_size)

    for i in range(0, seq_len, BLOCK_SIZE):
        for j in range(0, seq_len, BLOCK_SIZE):
            submask = torch.empty(
                (BLOCK_SIZE, BLOCK_SIZE), dtype=torch.bool, device=device
            )

            func_wrapper[grid](
                bitfield_mask,
                submask,
                i,
                i + BLOCK_SIZE,
                j,
                j + BLOCK_SIZE,
                0,
                0,
                BLOCK_SIZE,
            )
            torch.testing.assert_close(
                submask, expected_submask[i : i + BLOCK_SIZE, j : j + BLOCK_SIZE]
            )


@pytest.mark.parametrize("head_dim", [32, 64, 128], ids=lambda x: f"hd={x}")
@pytest.mark.parametrize(
    "seqlen", [57, 128, 144, 283, 256, 1024], ids=lambda x: f"sl={x}"
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
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), full_mask.unsqueeze(1)
    ).transpose(1, 2)

    # without materializing the full mask,
    # our Triton implementation accepts the bitfield mask directly
    # and internally materializes the mask in a tiled fashion
    triton_out = bitfield_attn_func(q, k, v, None, None, bitfield_mask)
    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(dq_ref, dq, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize(
    "offset, subset_size", [(24, 30), (77, 24), (0, 32), (200, 64)]
)
@pytest.mark.parametrize(
    "seqlen", [54, 57, 128, 144, 283, 256, 1024], ids=lambda x: f"sl={x}"
)
@pytest.mark.parametrize("batch_size", [1, 2, 4], ids=lambda x: f"b={x}")
def test_subquery_bitfield_attention(
    offset: int, subset_size: int, seqlen: int, batch_size: int
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 8
    head_dim = 64

    if offset + subset_size > seqlen:
        pytest.skip("Subset size exceeds sequence length")

    # torch.random.manual_seed(0)

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

    local_q = q[:, offset : offset + subset_size, :, :].contiguous()
    local_mask = full_mask[:, offset : offset + subset_size, :].contiguous()

    reference_out = sdpa(
        local_q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        local_mask.unsqueeze(1),
    ).transpose(1, 2)
    triton_out = bitfield_attn_func(
        local_q,
        k,
        v,
        None,
        None,
        bitfield_mask,
        None,
        None,
        torch.arange(offset, offset + subset_size, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .repeat(batch_size, 1),
    )

    torch.testing.assert_close(reference_out, triton_out, atol=5e-3, rtol=5e-3)

    full_g = torch.randn_like(q).normal_(mean=0, std=0.05)
    g = full_g[:, offset : offset + subset_size, :, :].contiguous()
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (local_q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (local_q, k, v), g)

    torch.testing.assert_close(dq_ref, dq, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(dk_ref, dk, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(dv_ref, dv, atol=5e-3, rtol=5e-3)

    # ========================================================================
    # Compare the result vs full attention computation
    # ========================================================================
    reference_out = sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), full_mask.unsqueeze(1)
    ).transpose(1, 2)
    torch.testing.assert_close(
        reference_out[:, offset : offset + subset_size, :, :],
        triton_out,
        atol=5e-3,
        rtol=5e-3,
    )
    dg_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), full_g)
    torch.testing.assert_close(
        dg_ref[:, offset : offset + subset_size, :, :], dq, atol=5e-3, rtol=5e-3
    )
    # compute other parts
    other_q = torch.cat(
        [q[:, :offset, :, :], q[:, offset + subset_size :, :, :]], dim=1
    )

    triton_other_out = bitfield_attn_func(
        other_q,
        k,
        v,
        None,
        None,
        bitfield_mask,
        None,
        None,
        torch.cat(
            [
                torch.arange(0, offset, dtype=torch.int32, device="cuda"),
                torch.arange(
                    offset + subset_size, seqlen, dtype=torch.int32, device="cuda"
                ),
            ],
            dim=0,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1),
    )

    other_g = torch.cat(
        [full_g[:, :offset, :, :], full_g[:, offset + subset_size :, :, :]], dim=1
    )

    _, odk, odv = torch.autograd.grad(triton_other_out, (other_q, k, v), other_g)

    torch.testing.assert_close(dk_ref, dk + odk, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(dv_ref, dv + odv, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "seqlens",
    [(57, 34), (128, 82), (144, 46), (283, 273), (256, 124), (1024, 133)]
    + [(34, 57), (82, 128), (46, 144), (273, 283), (124, 255), (133, 1024)],
    ids=lambda x: f"sl=({x[0]}, {x[1]})",
)
def test_bitfield_attention_with_batched_different_seqlens(seqlens: tuple[int]):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = len(seqlens)
    num_heads = 1
    head_dim = 64

    q = torch.randn(
        batch_size,
        max(seqlens),
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    ).normal_(mean=0, std=0.5)
    k = torch.randn(
        batch_size,
        max(seqlens),
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    ).normal_(mean=0, std=0.5)
    v = torch.randn(
        batch_size,
        max(seqlens),
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
    ).normal_(mean=0, std=0.5)

    for t in [q, k, v]:
        t.requires_grad_(True)

    # create a bitfield attention mask
    def get_bitfield_attention_mask() -> tuple[torch.Tensor, torch.Tensor]:
        text_bit = (1 << 62) | 1 | (1 << 1) | (1 << 2)

        attention_mask = torch.full(
            (batch_size, max(seqlens)),
            text_bit,
            dtype=torch.int64,
            device=device,
        )

        attention_mask[:, 12:24] = 1 << 1
        attention_mask[:, 36:56] = 1 << 2

        for batch_index, seqlen in enumerate(seqlens):
            attention_mask[batch_index, seqlen:] = 0

        full_mask = torch.tril(
            torch.ones(
                (batch_size, max(seqlens), max(seqlens)),
                dtype=torch.bool,
                device=device,
            ),
            diagonal=0,
        )
        full_mask[:, 12:24, :] = False
        full_mask[:, 12:24, 12:24] = True
        full_mask[:, 36:56, :] = False
        full_mask[:, 36:56, 36:56] = True

        for batch_index, seqlen in enumerate(seqlens):
            full_mask[batch_index, seqlen:, :] = False
            full_mask[batch_index, :, seqlen:] = False

        return attention_mask, full_mask

    with torch.no_grad():
        bitfield_mask, full_mask = get_bitfield_attention_mask()

    reference_out = sdpa(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), full_mask.unsqueeze(1)
    ).transpose(1, 2)

    triton_out = bitfield_attn_func(
        q,
        k,
        v,
        None,
        None,
        bitfield_mask,
        torch.tensor(list(seqlens), dtype=torch.int64, device="cuda"),
        torch.tensor(list(seqlens), dtype=torch.int64, device="cuda"),
        None,
    )

    full_g = torch.randn_like(q).normal_(mean=0, std=0.05)
    for batch_index, seqlen in enumerate(seqlens):
        full_g[batch_index, seqlen:] = 0

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), full_g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), full_g)

    # sdpa holds garbage data in the padded region
    for batch_index, seqlen in enumerate(seqlens):
        reference_out[batch_index, seqlen:] = torch.nan
    torch.testing.assert_close(
        reference_out, triton_out, atol=5e-3, rtol=5e-3, equal_nan=True
    )
    for batch_index, seqlen in enumerate(seqlens):
        assert triton_out[batch_index, seqlen:].isnan().all()

    torch.testing.assert_close(
        dq_ref.nan_to_num(), dq.nan_to_num(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        dk_ref.nan_to_num(), dk.nan_to_num(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(
        dv_ref.nan_to_num(), dv.nan_to_num(), atol=5e-3, rtol=5e-3
    )
