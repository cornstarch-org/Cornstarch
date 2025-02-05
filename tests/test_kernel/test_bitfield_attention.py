import math

import pytest
import torch
import triton
import triton.language as tl
from einops import rearrange, repeat

from cornstarch.kernel.attention import flash_attn_func, get_submask_from_bitfield_mask


def construct_local_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size: tuple[int, int] = (-1, -1),  # -1 means infinite window size
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
            attention_mask.expand(q.shape[0], q.shape[1], -1, -1) == 0,
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
@pytest.mark.parametrize("seqlen", [(256, 256), (1024, 256), (256, 1024), (1024, 1024)])
def test_attention(dtype: torch.dtype, head_dim: int, seqlen: tuple[int, int]):
    device = torch.device("cuda")
    batch_size = 2
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

    mask = torch.randint(0, 2, (seqlen_q, seqlen_k), device=device)

    reference_out = reference_attention(q, k, v, mask)
    triton_out = flash_attn_func(q, k, v, None, False, mask)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dq_ref, dq, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv, rtol=5e-3, atol=5e-3)


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


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seqlen", [128, 256, 1024])
def test_bitfield_attention(head_dim: int, seqlen: int):
    device = torch.device("cuda")
    dtype = torch.float16
    batch_size = 1
    num_heads = 12

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
    def get_bitfield_attention_mask() -> torch.Tensor:
        text_bit = (1 << 62) | 1 | (1 << 1) | (1 << 2)

        attention_mask = torch.full(
            (batch_size, seqlen),
            text_bit,
            dtype=torch.int64,
            device=device,
        )
        attention_mask[:, 12:24] = 1 << 1
        attention_mask[:, 36:56] = 1 << 2

        return attention_mask

    def convert_bitfield_attention_mask_to_full_mask(bitfield_mask: torch.Tensor):
        batch_size, seqlen = bitfield_mask.shape
        causal_mask = torch.tril(
            torch.ones((batch_size, seqlen, seqlen), dtype=torch.bool, device=device),
            diagonal=0,
        )

        q_bitfield_mask = bitfield_mask
        kv_bitfield_mask = bitfield_mask

        is_text_token = ((q_bitfield_mask & 1) > 0).unsqueeze(-1)

        q_modality_bits = (q_bitfield_mask & ((1 << 62) - 1)).unsqueeze(-1)
        kv_modality_bits = (kv_bitfield_mask & ((1 << 62) - 1)).unsqueeze(1)

        return (
            causal_mask
            & (is_text_token == True)
            & ((q_modality_bits & kv_modality_bits) > 0)
        ) | (
            (is_text_token == False)
            & (q_modality_bits == kv_modality_bits)
            & (q_bitfield_mask > 0).unsqueeze(-1)
        )

    mask = get_bitfield_attention_mask()

    full_mask = convert_bitfield_attention_mask_to_full_mask(mask)
    reference_out = reference_attention(q, k, v, full_mask[0])

    # without materializing the full mask,
    # our Triton implementation accepts the bitfield mask directly
    # and internally materializes the mask in a tiled fashion
    triton_out = flash_attn_func(q, k, v, None, False, mask)
    torch.testing.assert_close(reference_out, triton_out, rtol=5e-3, atol=5e-3)

    g = torch.randn_like(reference_out)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(reference_out, (q, k, v), g)
    dq, dk, dv = torch.autograd.grad(triton_out, (q, k, v), g)

    torch.testing.assert_close(dq_ref, dq, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dk_ref, dk, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(dv_ref, dv, rtol=5e-3, atol=5e-3)
