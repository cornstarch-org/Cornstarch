import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    _apply_kernel_options,
    _identity,
)
from torch._higher_order_ops.flex_attention import flex_attention as flex_attention_hop
from torch._higher_order_ops.flex_attention import (
    flex_attention_backward as flex_attention_hop_backward,
)
from torch._higher_order_ops.flex_attention import flex_attention_autograd
from cornstarch.kernel.interface import FlexAttnAnyMask
from cornstarch.kernel.interface import convert_attention_mask_to_block_mask
import pytest

from torch.testing import assert_close

torch._dynamo.config.cache_size_limit = 128


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM",
    [
        (1, 2, 128, 64),
        (1, 4, 128, 128),
        (1, 8, 1024, 128),
        (4, 7, 512, 128),
        (32, 15, 1024, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16]
)  # NOTE(runyu): "fp16", "fp32" is not supported yet
@pytest.mark.parametrize(
    "mask_type", ["random", "causal", "full"]
)  # NOTE(runyu): "causal" and "full" are not supported yet
def test_op_any_mask(Z, H, N_CTX, HEAD_DIM, dtype, mask_type):
    torch.manual_seed(20)
    q = torch.empty(
        (Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda"
    ).requires_grad_()
    k = torch.empty(
        (Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda"
    ).requires_grad_()
    v = torch.empty(
        (Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda"
    ).requires_grad_()
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale

    print(
        f"Batch size: {Z}, Head size: {H}, Context size: {N_CTX}, Head dim: {HEAD_DIM}, mask type: {mask_type}, dtype: {dtype}"
    )

    if mask_type == "random":
        # create random mask
        M = torch.randint(
            0, 2, (N_CTX, N_CTX), dtype=torch.uint8, device="cuda", requires_grad=False
        )
    elif mask_type == "causal":
        # M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        M = torch.ones((N_CTX, N_CTX), device="cuda", dtype=torch.uint8)
        for i in range(N_CTX):
            M[i, i + 1 :] = 0
    elif mask_type == "full":
        M = torch.ones((N_CTX, N_CTX), device="cuda", dtype=torch.uint8)

    mask = torch.broadcast_to(M, (Z, N_CTX, N_CTX))
    p[:, :, M == 0] = float("-inf")
    lse = torch.logsumexp(p.float(), dim=-1)
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    num_heads = q.shape[1]
    block_mask = convert_attention_mask_to_block_mask(mask, num_heads)

    # 1. flex_attention, passed
    # flex_out, softmax_lse = flex_attention(q, k, v, block_mask=block_mask, scale=sm_scale, return_lse=True)

    # 2. flex_attention_hop, passed
    # kernel_options = _apply_kernel_options(q, k, v, return_lse=True, kernel_options=None)
    # flex_out, softmax_lse = flex_attention_hop(q, k, v, block_mask=block_mask.as_tuple(), scale=sm_scale, score_mod=_identity, kernel_options=kernel_options)

    # 3. flex_attention_autograd, passed
    # kernel_options = _apply_kernel_options(q, k, v, return_lse=True, kernel_options=None)
    # flex_out, softmax_lse = flex_attention_autograd(q, k, v, score_mod=_identity, block_mask=block_mask.as_tuple(), scale=sm_scale, kernel_options=kernel_options)

    # 4. flex_attention_anymask of our own implementation, passed
    flex_out, softmax_lse, _, _ = FlexAttnAnyMask.apply(
        q, k, v, 0.0, sm_scale, mask, -1, -1, 0.0, None, False, True
    )

    flex_out.backward(dout)
    flex_dv, v.grad = v.grad.clone(), None
    flex_dk, k.grad = k.grad.clone(), None
    flex_dq, q.grad = q.grad.clone(), None

    atol = 1e-3
    rtol = 1e-3

    assert_close(flex_out, ref_out, atol=atol, rtol=rtol)
    assert_close(flex_dv, ref_dv, atol=atol, rtol=rtol)
    assert_close(flex_dk, ref_dk, atol=atol, rtol=rtol)
    assert_close(flex_dq, ref_dq, atol=atol, rtol=rtol)
    assert_close(softmax_lse, lse, atol=atol, rtol=rtol)

    print(
        f"test passed for Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, mask_type={mask_type}, dtype={dtype}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_op_any_mask(5, 7, 2048, 128, torch.bfloat16, "random")
