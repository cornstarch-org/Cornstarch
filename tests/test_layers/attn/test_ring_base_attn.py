import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_func
from cornstarch.shardformer.layers.attn import RingAttentionBase

def set_seed(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_test(rank, world_size, batch_size, seq_len, num_heads, head_dim):
    # Initialize process group
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_seed(rank)

    rtol = atol = 7e-3
    
    # Set up input tensors
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16

    qkv = torch.randn(batch_size, seq_len, 3, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    dist.broadcast(qkv, src=0)

    dout = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    dist.broadcast(dout, src=0)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True
    local_dout = dout.chunk(world_size, dim=1)[rank].detach().clone()

    q, k, v = qkv.chunk(3, dim=2)
    q = q.squeeze(dim=2).detach().clone()
    k = k.squeeze(dim=2).detach().clone()
    v = v.squeeze(dim=2).detach().clone()
    assert q.shape == torch.Size([batch_size, seq_len, num_heads, head_dim]), f"q shape is {q.shape}"
    assert k.shape == torch.Size([batch_size, seq_len, num_heads, head_dim]), f"k shape is {k.shape}"
    assert v.shape == torch.Size([batch_size, seq_len, num_heads, head_dim]), f"v shape is {v.shape}"
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    # Run flash_attn_func
    out, lse, _ = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    # run RingAttentionBase
    local_q, local_k, local_v = local_qkv.chunk(3, dim=2)
    local_q = local_q.squeeze(dim=2).detach().clone()
    local_k = local_k.squeeze(dim=2).detach().clone()
    local_v = local_v.squeeze(dim=2).detach().clone()
    assert local_q.shape == torch.Size([batch_size, seq_len // world_size, num_heads, head_dim]), f"q shape is {local_q.shape}"
    assert local_k.shape == torch.Size([batch_size, seq_len // world_size, num_heads, head_dim]), f"k shape is {local_k.shape}"
    assert local_v.shape == torch.Size([batch_size, seq_len // world_size, num_heads, head_dim]), f"v shape is {local_v.shape}"
    local_q.requires_grad = True
    local_k.requires_grad = True
    local_v.requires_grad = True
    ring_out, ring_lse = RingAttentionBase.apply(
        local_q,
        local_k,
        local_v,
        dist.group.WORLD,
        True, # causal
        True, # return_softmax
        0.0, # dropout_p
        None, # softmax_scale
        False, # deterministic
        -1, # window_size_left
        -1, # window_size_right
        None, # alibi_slopes
    )

    # Compare outputs
    assert torch.allclose(local_out, ring_out, rtol=rtol, atol=atol), \
        f"ring_flash_attn_func output does not match flash_attn_func output on rank {rank}"
    assert torch.allclose(local_lse, ring_lse, rtol=rtol, atol=atol), \
        f"ring_flash_attn_func LSE does not match flash_attn_func LSE on rank {rank}"

    # Backward pass
    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = dq.chunk(world_size, dim=1)[rank]
    local_dk = dk.chunk(world_size, dim=1)[rank]
    local_dv = dv.chunk(world_size, dim=1)[rank]
    # dqkv = qkv.grad
    # local_dqkv = dqkv.chunk(world_size, dim=1)[rank]

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad
    # ring_dqkv = local_qkv.grad

    # Compare gradients
    assert torch.allclose(local_dq, ring_dq, rtol=rtol, atol=atol), \
        f"ring_flash_attn_qkvpacked_func gradient does not match flash_attn_qkvpacked_func gradient on rank {rank}"
    assert torch.allclose(local_dk, ring_dk, rtol=rtol, atol=atol), \
        f"ring_flash_attn_qkvpacked_func gradient does not match flash_attn_qkvpacked_func gradient on rank {rank}"
    assert torch.allclose(local_dv, ring_dv, rtol=rtol, atol=atol), \
        f"ring_flash_attn_qkvpacked_func gradient does not match flash_attn_qkvpacked_func gradient on rank {rank}"

    print(f"Test passed on rank {rank}: ring_flash_attn_qkvpacked_func matches flash_attn_qkvpacked_func")

    # Clean up
    dist.destroy_process_group()

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_heads", [5])
@pytest.mark.parametrize("head_dim", [128])
# @pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("world_size", [2])
def test_ring_flash_attn_vs_flash_attn(batch_size, seq_len, num_heads, head_dim, world_size):
    assert seq_len % world_size == 0, "seq_len must be divisible by world_size"
    assert head_dim % 8 == 0, "head_dim must be divisible by 8"

    mp.spawn(
        run_test,
        args=(world_size, batch_size, seq_len, num_heads, head_dim),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    pytest.main([__file__])
