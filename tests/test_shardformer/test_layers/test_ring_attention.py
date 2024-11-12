from typing import Callable

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.kernel.interface import flash_attn_triton_func
from cornstarch.shardformer.layers.ring_attention import RingAttentionFixedlen

from .utils import RingAttentionTestBase


@instantiate_parametrized_tests
class TestRingAttentionFixedlenClass(RingAttentionTestBase):
    """
    A class to test the RingAttentionBase class.
    RingAttentionBase provides implementation for basic
    distributed full bi-directional or causal attention mechanism
    with ring-topology communication.
    """

    @parametrize("kernel_impl", ["cuda", "triton"], lambda x: x)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("causal", [True, False])
    @parametrize("world_size", [4, 8])
    def test(self, kernel_impl: str, dtype: torch.dtype, causal: bool, world_size: int):
        rtol = atol = 7e-3

        attention_func: Callable
        seq_dim: int
        if kernel_impl == "cuda":
            attention_func = flash_attn_func
            seq_dim = 1
        else:
            attention_func = flash_attn_triton_func
            seq_dim = 2

        q, k, v, dout = self.prepare_qkv_for_flash_attention(kernel_impl, dtype)

        local_q, local_k, local_v, local_dout = self.prepare_qkv_for_ring_attention(
            (q, k, v), dout, seq_dim
        )

        # Check inputs
        self.check_tensors(
            reference_tensors=(
                q.chunk(world_size, dim=seq_dim)[self.rank],
                k.chunk(world_size, dim=seq_dim)[self.rank],
                v.chunk(world_size, dim=seq_dim)[self.rank],
            ),
            test_tensors=(local_q, local_k, local_v),
            rtol=rtol,
            atol=atol,
        )

        # out:
        #   with cuda kernel: [batch_size, seq_len, num_heads, head_dim]
        #   with triton kernel: [batch_size, num_heads, seq_len, head_dim]
        # lse: [batch_size, num_heads, seq_len]
        out, lse, _ = attention_func(
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

        # ring_out:
        #   with cuda kernel: [batch_size, seq_len // world_size, num_heads, head_dim]
        #   with triton kernel: [batch_size, num_heads, seq_len // world_size, head_dim]
        # ring_lse:
        #   with cuda kernel: [batch_size, seq_len // world_size, num_heads]
        #   with triton kernel: [batch_size, num_heads, seq_len // world_size]
        ring_out, ring_lse = RingAttentionFixedlen.apply(
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

        # Check outputs
        self.check_tensors(
            reference_tensors=(
                out.chunk(world_size, dim=seq_dim)[self.rank],
                (
                    sharded_lse.transpose(-1, -2)
                    if (sharded_lse := lse.chunk(world_size, dim=-1)[self.rank]).shape[
                        :3
                    ]
                    != ring_lse.shape[:3]
                    else sharded_lse
                ),
            ),
            test_tensors=(ring_out, ring_lse),
            rtol=rtol,
            atol=atol,
        )

        # Backward pass
        out.backward(dout)
        ring_out.backward(local_dout)

        # Compare gradients
        self.check_tensors(
            reference_tensors=(
                q.grad.chunk(world_size, dim=seq_dim)[self.rank],
                k.grad.chunk(world_size, dim=seq_dim)[self.rank],
                v.grad.chunk(world_size, dim=seq_dim)[self.rank],
            ),
            test_tensors=(local_q.grad, local_k.grad, local_v.grad),
            rtol=rtol,
            atol=atol,
        )
