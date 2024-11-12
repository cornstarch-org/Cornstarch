import torch
import pytest
from cornstarch.kernel.interface import FlashAttnAnyMask

from torch.testing import assert_close


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64), (1, 4, 1024, 128), (1, 8, 1024, 128)]
)
@pytest.mark.parametrize(
    "causal", [True]
)  # TODO(@runyu): add False, it is werid that bwd of triton flash attention don't support causal=False
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_op_casual(Z, H, N_CTX, HEAD_DIM, causal, dtype):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    if causal:
        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    else:
        M = torch.ones((N_CTX, N_CTX), device="cuda", dtype=torch.uint8)
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    # tri_out = flash_attn_triton_func(q, k, v, causal=causal, softmax_scale=sm_scale).to(dtype)
    mask = torch.broadcast_to(M, (Z, H, N_CTX, N_CTX))
    tri_out, softmax_lse = FlashAttnAnyMask.apply(q, k, v, mask, sm_scale, 0.0)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    atol = 7e-3
    rtol = 1e-2

    # compare
    assert_close(ref_out, tri_out, atol=atol, rtol=rtol)
    assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM",
    [(1, 2, 128, 64), (1, 4, 128, 128), (1, 8, 1024, 128), (4, 7, 512, 128)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask_type", ["random", "causal", "full"])
def test_op_any_mask(Z, H, N_CTX, HEAD_DIM, dtype, mask_type):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
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
    mask = torch.broadcast_to(M, (Z, H, N_CTX, N_CTX))
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    tri_out, softmax_lse = FlashAttnAnyMask.apply(q, k, v, mask, sm_scale, 0.0)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None

    atol = 7e-3
    rtol = 1e-2

    # compare
    assert_close(ref_out, tri_out, atol=atol, rtol=rtol)
    assert_close(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert_close(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert_close(ref_dq, tri_dq, atol=atol, rtol=rtol)

    print(
        f"test passed for Z={Z}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, mask_type={mask_type}, dtype={dtype}"
    )


if __name__ == "__main__":
    pytest.main([__file__])
    # test_op_casual(1, 2, 1024, 64, causal=True, dtype=torch.bfloat16)
    # test_op_casual(4, 5, 1024, 128, causal=True, dtype=torch.float16)
    # test_op_casual(1, 2, 1024, 64, causal=False, dtype=torch.bfloat16)
    # test_op_any_mask(1, 2, 128, 64, dtype=torch.bfloat16, mask_type="causal")
    # test_op_any_mask(4, 5, 512, 128, dtype=torch.bfloat16, mask_type="causal")
    # test_op_any_mask(1, 2, 128, 64, dtype=torch.bfloat16, mask_type="random")
    # test_op_any_mask(1, 2, 1024, 128, dtype=torch.bfloat16, mask_type="random")
    # test_op_any_mask(1, 2, 128, 128, dtype=torch.bfloat16, mask_type="random")
    # test_op_any_mask(1, 2, 1024, 128, dtype=torch.bfloat16, mask_type="full")
    # test_op_any_mask(1, 2, 1024, 128, dtype=torch.bfloat16, mask_type="full")
