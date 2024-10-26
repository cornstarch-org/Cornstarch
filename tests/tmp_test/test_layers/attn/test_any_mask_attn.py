import pytest
import torch
from cornstarch.kernel.interface import _flash_attn_anymask_forward, _flash_attn_anymask_backward
from cornstarch.kernel.interface import FlashAttnAnyMask

from torch.testing import assert_close

def set_seed(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def refer_attn_with_mask(q, k, v, mask, sm_scale):
    qk = q @ k.transpose(-2, -1)
    if mask is not None:
        # qk = qk * mask + (1 - mask) * -1e9
        qk = qk * mask
    
    qk = qk * sm_scale
    
    # Calculate max for numerical stability
    max_qk = torch.max(qk, dim=-1, keepdim=True)[0]
    
    # Subtract max and apply exp
    exp_qk = torch.exp(qk - max_qk)
    
    # Apply mask to exp_qk
    if mask is not None:
        exp_qk = exp_qk * mask
    
    # Calculate sum for softmax denominator and LSE
    sum_exp_qk = torch.sum(exp_qk, dim=-1, keepdim=True)
    lse = torch.log(sum_exp_qk) + max_qk
    
    # Calculate softmax
    softmax_qk = exp_qk / sum_exp_qk
    
    out = softmax_qk.to(q.dtype) @ v
    return out, lse.squeeze(-1)

# @pytest.mark.parametrize("batch_size", [4])
# @pytest.mark.parametrize("seq_len", [256])
# @pytest.mark.parametrize("num_heads", [5])
# @pytest.mark.parametrize("head_dim", [64])
# @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
# @pytest.mark.parametrize("seq_len", [1024, 256])
# @pytest.mark.parametrize("num_heads", [5, 8])
# @pytest.mark.parametrize("head_dim", [128, 64])
# def test_run(batch_size, seq_len, num_heads, head_dim):
#     dtype = torch.float16
#     # dtype = torch.float32
#     device = torch.device("cuda")
#     set_seed(2)
#     q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
#     k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
#     v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_()
#     q.requires_grad = True
#     k.requires_grad = True
#     v.requires_grad = True

#     # Create a binary mask (0 and 1) with dtype=int8
#     M = mask = torch.randint(0, 2, (seq_len, seq_len), dtype=torch.int8, device=device)
#     mask = torch.broadcast_to(mask, (batch_size, num_heads, seq_len, seq_len))
#     sm_scale = 1.0 / (head_dim ** 0.5)

#     # forward
#     out, softmax_lse = FlashAttnAnyMask.apply(q, k, v, mask, sm_scale, 0.0)

#     ref_q = q.clone().detach()
#     ref_k = k.clone().detach()
#     ref_v = v.clone().detach()
#     ref_q.requires_grad = True
#     ref_k.requires_grad = True
#     ref_v.requires_grad = True

#     # reference implementation
#     p = torch.matmul(ref_q, ref_k.transpose(2, 3))
#     p[:, :, M == 0] = float("-inf")
#     # p = p * sm_scale
#     # compute lse
#     ref_lse = torch.max(p, dim=-1, keepdim=True)[0].squeeze(-1) + torch.log(torch.sum(torch.exp(p.float() - torch.max(p.float(), dim=-1, keepdim=True)[0]), dim=-1, keepdim=True)).squeeze(-1)
#     p = p * sm_scale
#     p = torch.softmax(p.float(), dim=-1).to(ref_v.dtype)
#     ref_out = torch.matmul(p, ref_v)

#     # from flash_attn.flash_attn_interface import flash_attn_func
#     # ref_out, ref_lse, _ = flash_attn_func(ref_q.transpose(1, 2), ref_k.transpose(1, 2), ref_v.transpose(1, 2), sm_scale, causal=False, window_size=(-1, -1), softcap=0.0, alibi_slopes=None, deterministic=False, return_attn_probs=True)
#     # ref_out = ref_out.transpose(1, 2)
#     # ref_lse = ref_lse.transpose(1, 2)

#     atol = rtol = 7e-3
#     assert torch.allclose(out, ref_out, atol=atol, rtol=rtol)

#     # TODO(runyu): Although lse and bwd are not correct, but there are some multiple relationships between the values.
#     assert softmax_lse.shape == ref_lse.shape, f"softmax_lse.shape: {softmax_lse.shape}, ref_lse.shape: {ref_lse.shape}"
#     assert torch.allclose(softmax_lse, ref_lse, atol=atol, rtol=rtol) # TODO(runyu): fix this

#     # backward # TODO(runyu): fix this, it is strange that even mask is filled with 1, the gradients are not the same, and some values are same, some are not.
#     dout = torch.randn_like(out)

#     ref_out.backward(dout)

#     out.backward(dout)

#     atol, rtol = 7e-3, 7e-3
#     dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
#     ref_dv, ref_dk, ref_dq = ref_v.grad.clone(), ref_k.grad.clone(), ref_q.grad.clone()

#     assert isinstance(dq, torch.Tensor)
#     assert isinstance(ref_dq, torch.Tensor)

#     assert_close(dq, ref_dq, atol=atol, rtol=rtol)
#     assert_close(dk, ref_dk, atol=atol, rtol=rtol)
#     assert_close(dv, ref_dv, atol=atol, rtol=rtol)

def softmax_base2_to_e_with_scale(x, scale=1.0, dim=-1):
    # Apply scale
    x = x * scale * 1.44269504 # 1/log(2)
    
    # Compute base-2 softmax
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - x_max
    exp_x = torch.exp2(x)
    sum_exp_x = torch.sum(exp_x.float(), dim=dim, keepdim=True)
    softmax_2 = exp_x / sum_exp_x

    lse = torch.log2(sum_exp_x.float()).squeeze(dim) + x_max.float().squeeze(dim)

    return softmax_2, lse
    
    # Convert to e-based softmax
    # log_2 = torch.log(torch.tensor(2.0))
    # return softmax_2.pow(log_2)

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 128, 64), (1, 4, 128, 128), (1, 8, 1024, 128), (4, 7, 512, 128)])
def test_op_random(Z, H, N_CTX, HEAD_DIM, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    
    # create random mask
    M = mask = torch.randint(0, 2, (N_CTX, N_CTX), dtype=torch.uint8, device="cuda", requires_grad=False)
    mask = torch.broadcast_to(mask, (Z, H, N_CTX, N_CTX))
    p[:, :, M == 0] = float("-inf")
    _, ref_softmax_lse = softmax_base2_to_e_with_scale(p, scale=sm_scale)
    p = torch.softmax(p.float(), dim=-1).half()
    # ref_softmax_lse = torch.max(p.float(), dim=-1, keepdim=True)[0].squeeze(-1) + torch.log(torch.sum(torch.exp(p.float() - torch.max(p.float(), dim=-1, keepdim=True)[0]), dim=-1, keepdim=True)).squeeze(-1)
    # p = torch.exp(p)
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
    assert torch.allclose(ref_out, tri_out, atol=atol, rtol=rtol)
    # assert torch.allclose(ref_softmax_lse, softmax_lse, atol=1e-3, rtol=1e-3)
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    # if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        # rtol = 1e-2
    
    assert torch.allclose(ref_dv, tri_dv, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=atol, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__])
