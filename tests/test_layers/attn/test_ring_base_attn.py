import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_func
from torch.testing import assert_close

from cornstarch.kernel.interface import flash_attn_triton_func
from cornstarch.shardformer.layers.attn import RingAttentionBase


def set_seed(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_qkv(
    rank,
    world_size,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    kernel_impl,
    device,
    dtype,
):
    """
    prepare q, k, v tensor for flash attn, local_q, local_k, local_v for ring attention
    """

    if kernel_impl == "cuda":
        seq_dim = 1
    elif kernel_impl == "triton":
        seq_dim = 2
    else:
        raise ValueError(
            f"kernel_impl must be either cuda or triton, but got {kernel_impl}"
        )

    qkv = (
        torch.randn(
            3, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )
    dist.broadcast(qkv, src=0)

    dout = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_(True)
    )
    dist.broadcast(
        dout, src=0
    )  # now each rank has the same qkv tensor and a dout tensor

    if kernel_impl == "triton":
        qkv = qkv.transpose(
            2, 3
        )  # triton requires q, k, v to be in shape [batch_size, num_heads, seq_len, head_dim]
        dout = dout.transpose(1, 2)

    dout = dout.contiguous()

    local_qkv = (
        qkv.chunk(world_size, dim=seq_dim + 1)[rank]
        .detach()
        .clone()
        .contiguous()
        .requires_grad_(True)
    )
    local_dout = (
        dout.chunk(world_size, dim=seq_dim)[rank]
        .detach()
        .clone()
        .contiguous()
        .requires_grad_(True)
    )  # now each rank has the same local_qkv and local_dout

    # split q, k, v and local_q, local_k, local_v
    q, k, v = qkv.chunk(3, dim=0)
    q = q.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)
    k = k.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)
    v = v.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)

    local_q, local_k, local_v = local_qkv.chunk(3, dim=0)
    local_q = local_q.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)
    local_k = local_k.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)
    local_v = local_v.squeeze(dim=0).detach().clone().contiguous().requires_grad_(True)

    assert (
        torch.allclose(q.chunk(world_size, dim=seq_dim)[rank], local_q)
        and torch.allclose(k.chunk(world_size, dim=seq_dim)[rank], local_k)
        and torch.allclose(v.chunk(world_size, dim=seq_dim)[rank], local_v)
    )

    return q, k, v, dout, local_q, local_k, local_v, local_dout


def run_test(
    rank, world_size, batch_size, seq_len, num_heads, head_dim, kernel_impl, causal=True
):
    # Initialize process group
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_seed(rank)

    rtol = atol = 7e-3

    if dist.get_rank() == 0:
        print(
            f"Running test with batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}, world_size={world_size}, kernel_impl={kernel_impl}, causal={causal}"
        )

    # Set up input tensors
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    # dtype = torch.float16

    if kernel_impl == "cuda":
        attn_func = flash_attn_func
        seq_dim = 1
    elif kernel_impl == "triton":
        attn_func = flash_attn_triton_func
        seq_dim = 2
    else:
        raise ValueError(
            f"kernel_impl must be either cuda or triton, but got {kernel_impl}"
        )

    q, k, v, dout, local_q, local_k, local_v, local_dout = prepare_qkv(
        rank,
        world_size,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        kernel_impl,
        device,
        dtype,
    )
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    local_q.retain_grad()
    local_k.retain_grad()
    local_v.retain_grad()
    assert torch.allclose(q.chunk(world_size, dim=seq_dim)[rank], local_q)

    out, lse, _ = attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=True,
    )

    refer_local_out = out.chunk(world_size, dim=seq_dim)[
        rank
    ]  # [batch_size, num_heads, seq_len // world_size, head_dim]
    refer_local_lse = lse.chunk(world_size, dim=-1)[
        rank
    ]  # [batch_size, num_heads, seq_len // world_size]

    ring_out, ring_lse = RingAttentionBase.apply(
        local_q,
        local_k,
        local_v,
        dist.group.WORLD,
        causal,  # causal
        True,  # return_softmax
        0.0,  # dropout_p
        None,  # softmax_scale
        False,  # deterministic
        -1,  # window_size_left
        -1,  # window_size_right
        None,  # alibi_slopes
        kernel_impl,  # kernel_impl
    )

    # Compare outputs
    if refer_local_lse.shape[:3] != ring_lse.shape[:3]:
        refer_local_lse = refer_local_lse.transpose(-1, -2)
    assert_close(
        refer_local_out, ring_out, rtol=rtol, atol=atol
    ), f"{kernel_impl} ring_flash_attn_func output does not match flash_attn_func output on rank {rank}"
    assert_close(
        refer_local_lse, ring_lse, rtol=rtol, atol=atol
    ), f"{kernel_impl} ring_flash_attn_func LSE does not match flash_attn_func LSE on rank {rank}"

    # if dist.get_rank() == 0:
    #     if kernel_impl == "triton":
    #         print(f"ring_out shape: {ring_out.shape}, ring_out: {ring_out.transpose(1, 2)}")
    #         print(f"ring_lse shape: {ring_lse.shape}, ring_lse: {ring_lse}")
    #     else:
    #         print(f"ring_out shape: {ring_out.shape}, ring_out: {ring_out}")
    #         print(f"ring_lse shape: {ring_lse.shape}, ring_lse: {ring_lse}")

    print(
        f"Test passed on rank {rank}: {kernel_impl} ring_flash_attn_func matches flash_attn_func"
    )

    # Backward pass
    out.backward(dout)
    dq = q.grad
    dk = k.grad
    dv = v.grad
    local_dq = dq.chunk(world_size, dim=seq_dim)[rank]
    local_dk = dk.chunk(world_size, dim=seq_dim)[rank]
    local_dv = dv.chunk(world_size, dim=seq_dim)[rank]

    ring_out.backward(local_dout)
    ring_dq = local_q.grad
    ring_dk = local_k.grad
    ring_dv = local_v.grad

    assert_close(
        local_dq, ring_dq, rtol=rtol, atol=atol
    ), f"{kernel_impl} ring_flash_attn_func gradient does not match flash_attn_func gradient on rank {rank}"
    assert_close(
        local_dk, ring_dk, rtol=rtol, atol=atol
    ), f"{kernel_impl} ring_flash_attn_func gradient does not match flash_attn_func gradient on rank {rank}"
    assert_close(
        local_dv, ring_dv, rtol=rtol, atol=atol
    ), f"{kernel_impl} ring_flash_attn_func gradient does not match flash_attn_func gradient on rank {rank}"

    print(
        f"Backward test passed on rank {rank}: {kernel_impl} ring_flash_attn_func matches flash_attn_func"
    )

    # if dist.get_rank() == 0:
    #     if kernel_impl == "triton":
    #         print(f"local_q.grad shape: {local_q.grad.shape}, local_q.grad: {local_q.grad.transpose(1, 2)}")
    #         print(f"local_k.grad shape: {local_k.grad.shape}, local_k.grad: {local_k.grad.transpose(1, 2)}")
    #         print(f"local_v.grad shape: {local_v.grad.shape}, local_v.grad: {local_v.grad.transpose(1, 2)}")
    #     else:
    #         print(f"local_q.grad shape: {local_q.grad.shape}, local_q.grad: {local_q.grad}")
    #         print(f"local_k.grad shape: {local_k.grad.shape}, local_k.grad: {local_k.grad}")
    #         print(f"local_v.grad shape: {local_v.grad.shape}, local_v.grad: {local_v.grad}")

    # Clean up
    dist.destroy_process_group()


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize("num_heads", [5])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("kernel_impl", ["cuda", "triton"])
def test_ring_flash_attn_vs_flash_attn(
    batch_size, seq_len, num_heads, head_dim, world_size, kernel_impl
):
    assert seq_len % world_size == 0, "seq_len must be divisible by world_size"
    assert head_dim % 8 == 0, "head_dim must be divisible by 8"
    assert kernel_impl in [
        "cuda",
        "triton",
    ], "kernel_impl must be either cuda or triton"

    mp.spawn(
        run_test,
        args=(world_size, batch_size, seq_len, num_heads, head_dim, kernel_impl),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # pytest.main([__file__])

    batch_size = 4
    seq_len = 512
    num_heads = 5
    head_dim = 128
    world_size = 4
    kernel_impl = "triton"
    # kernel_impl = "cuda"
    causal = False
    # causal = True
    mp.spawn(
        run_test,
        args=(
            world_size,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            kernel_impl,
            causal,
        ),
        nprocs=world_size,
        join=True,
    )
