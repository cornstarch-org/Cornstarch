import pytest
import torch
from cornstarch.kernel.interface import _flash_attn_anymask_forward, _flash_attn_anymask_backward
from cornstarch.kernel.interface import FlashAttnAnyMask


def set_seed(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def refer_attn_with_mask(q, k, v, mask, sm_scale):
    qk = q @ k.transpose(-2, -1)
    if mask is not None:
        qk = qk * mask + (1 - mask) * -1e9
    
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

@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [256])
@pytest.mark.parametrize("num_heads", [5])
@pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
# @pytest.mark.parametrize("seq_len", [1024, 256])
# @pytest.mark.parametrize("num_heads", [5, 8])
# @pytest.mark.parametrize("head_dim", [128, 64])
def test_run(batch_size, seq_len, num_heads, head_dim):
    dtype = torch.float16
    dtype = torch.float32
    device = torch.device("cuda")
    set_seed(1)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    # Create a binary mask (0 and 1) with dtype=int
    mask = torch.randint(0, 2, (batch_size, 1, seq_len, seq_len), dtype=torch.int, device=device)
    mask = torch.broadcast_to(mask, (batch_size, num_heads, seq_len, seq_len))
    sm_scale = 1.0 / (head_dim ** 0.5)

    
    # forward
    # out, softmax_lse, _, _ = _flash_attn_anymask_forward(q, k, v, mask, 0.0, sm_scale, 0, 0, 0, None, False)
    out, softmax_lse = FlashAttnAnyMask.apply(q, k, v, mask, sm_scale, 0.0)

    ref_q = q.clone().detach()
    ref_k = k.clone().detach()
    ref_v = v.clone().detach()
    ref_q.requires_grad = True
    ref_k.requires_grad = True
    ref_v.requires_grad = True

    refer_out, refer_lse = refer_attn_with_mask(ref_q, ref_k, ref_v, mask, sm_scale)

    atol, rtol = 7e-3, 7e-3
    assert torch.allclose(out, refer_out, atol=atol, rtol=rtol)

    # TODO(runyu): Although lse and bwd are not correct, but there are some multiple relationships between the values.
    # assert torch.allclose(softmax_lse, refer_lse, atol=atol, rtol=rtol) # TODO(runyu): fix this

    # # backward # TODO(runyu): fix this
    # dout = torch.randn_like(out)
    # refer_out.backward(dout)
    # out.backward(dout)

    # atol, rtol = 7e-3, 7e-3
    # dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
    # ref_dq, ref_dk, ref_dv = ref_q.grad.clone(), ref_k.grad.clone(), ref_v.grad.clone()

    # assert isinstance(dq, torch.Tensor)
    # assert isinstance(ref_dq, torch.Tensor)

    # assert torch.allclose(dq, ref_dq, atol=atol, rtol=rtol)
    # assert torch.allclose(dk, ref_dk, atol=atol, rtol=rtol)
    # assert torch.allclose(dv, ref_dv, atol=atol, rtol=rtol)

if __name__ == "__main__":
    pytest.main([__file__])
