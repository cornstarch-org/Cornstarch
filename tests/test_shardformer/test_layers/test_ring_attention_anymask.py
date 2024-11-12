import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

from cornstarch.shardformer.layers.ring_attention_anymask import RingAttentionAnyMask

from .utils import RingAttentionTestBase


@instantiate_parametrized_tests
class TestRingAttentionAnymaskClass(RingAttentionTestBase):
    """
    A class to test the AnyMask RingAttention implementation.
    AnyMask accepts arbitrary forms of attention masks, other than
    just causal or full attention masks.
    """

    def prepare_attention_mask(self, mask_type: str) -> torch.Tensor:
        """
        Returns a 4D attention mask tensor with the shape of
        [batch_size, num_heads, seq_len, seq_len]
        """
        if mask_type == "any":
            mask = torch.randint(
                0,
                2,
                (self.batch_size, self.seq_len, self.seq_len),
                dtype=torch.uint8,
                device="cuda",
                requires_grad=False,
            )
        elif mask_type == "causal":
            mask = torch.tril(
                torch.ones(
                    (self.seq_len, self.seq_len), dtype=torch.uint8, device="cuda"
                )
            )
            mask = mask.unsqueeze(dim=0).expand(self.batch_size, -1, -1)
        elif mask_type == "full":
            mask = torch.ones(
                (self.batch_size, self.seq_len, self.seq_len),
                dtype=torch.uint8,
                device="cuda",
            )

        # Replicate mask to all heads
        mask = mask.unsqueeze(dim=1).expand(-1, self.num_heads, -1, -1).contiguous()
        dist.broadcast(mask, src=0)

        return mask

    def reference_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sm_scale: float,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention using the reference implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        # p[:, :, mask == 0] = float("-inf")
        p[mask == 0] = -1e4 if q.dtype == torch.float16 else -1e9
        lse = torch.logsumexp(p.float(), dim=-1)
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        out = torch.matmul(p, v)
        return out, lse

    @parametrize("kernel_impl", ["triton"], name_fn=lambda x: x)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("mask_type", ["any", "causal", "full"], name_fn=lambda x: x)
    @parametrize("world_size", [4, 8])
    def test(
        self, kernel_impl: str, dtype: torch.dtype, mask_type: str, world_size: int
    ):
        rtol = atol = 5e-2
        seq_dim = 2  # triton is always used

        q, k, v, dout = self.prepare_qkv_for_flash_attention(
            kernel_impl,
            dtype,
        )
        mask = self.prepare_attention_mask(mask_type)

        local_q, local_k, local_v, local_dout = self.prepare_qkv_for_ring_attention(
            (q, k, v), dout, seq_dim
        )
        local_mask = (
            mask.chunk(world_size, dim=seq_dim)[self.rank]
            .detach()
            .clone()
            .contiguous()
            .requires_grad_(False)
        )

        # Compare inputs
        self.check_tensors(
            reference_tensors=(
                q.chunk(world_size, dim=seq_dim)[self.rank],
                k.chunk(world_size, dim=seq_dim)[self.rank],
                v.chunk(world_size, dim=seq_dim)[self.rank],
                mask.chunk(world_size, dim=seq_dim)[self.rank],
            ),
            test_tensors=(local_q, local_k, local_v, local_mask),
        )

        out, lse = self.reference_attention(q, k, v, sm_scale=0.5, mask=mask)

        ring_out, ring_lse = RingAttentionAnyMask.apply(
            local_q,
            local_k,
            local_v,
            local_mask,
            dist.group.WORLD,
            True,  # return_softmax
            0.0,  # dropout_p
            0.5,  # softmax_scale
            False,  # deterministic
            -1,  # window_size_left
            -1,  # window_size_right
            None,  # alibi_slopes
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
