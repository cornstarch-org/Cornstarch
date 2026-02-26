from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed.fake_pg import FakeStore

from flash_attn.bert_padding import index_first_axis
from transformers.modeling_flash_attention_utils import _get_unpad_data

from cornstarch.kernel.bitfield_attention import BitfieldUtils, bitfield_attn_func
from cornstarch.shardformer.layers.context_parallel_attention import (
    ContextParallelFlashAttention,
)
from cornstarch.shardformer.layers.context_parallel_bitfield_attention import (
    ContextParallelBitfieldAttention,
)
from cornstarch.shardformer.layers.context_parallel_ring_attention import (
    ContextParallelVarlenRingAttention,
)
from cornstarch.shardformer.layers.utils import (
    ContextParallelBatchSplitUtils,
)
from cornstarch.shardformer.modeling.llama import _zigzag_local_indices

from ..distributed_base import GlooDistributedTestBase


def get_causal_assignments(num_blocks: int):
    """
    Assignment pattern for causal mask looks like:
    [..., 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]
    """
    base = [0, 1, 2, 3]
    tail = [3, 2, 1, 0]

    assignments = tail[:] if num_blocks >= 4 else tail[:num_blocks]
    current_length = len(assignments)
    toggle = True
    while current_length < num_blocks:
        needed = num_blocks - current_length
        prepend = base[:] if toggle else tail[:]
        if needed < 4:
            prepend = prepend[-needed:]
        assignments = prepend + assignments
        current_length = len(assignments)
        toggle = not toggle

    assignments = torch.as_tensor(assignments, device="cuda").repeat_interleave(128)

    return assignments


def get_full_assignments(num_blocks: int):
    pattern = [0, 1, 2, 3]
    assignments = (
        pattern * (num_blocks // len(pattern)) + pattern[: num_blocks % len(pattern)]
    )

    assignments = torch.as_tensor(assignments, device="cuda").repeat_interleave(128)

    return assignments


class TestContextParallelBatchSplitUtilClass:

    num_heads: int
    head_dim: int

    @classmethod
    def setup_class(cls: TestContextParallelBatchSplitUtilClass):
        cls.num_heads = 8
        cls.head_dim = 64

    @pytest.fixture(autouse=True)
    def cleanup(self):
        self.teardown_method()

    def teardown_method(self):
        if dist.is_initialized():
            dist.destroy_process_group()

        BitfieldUtils.clear_cache()
        ContextParallelBatchSplitUtils.clear_cache()

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_uniform_singlebatch(self, seqlen: int, world_size: int):
        batch_size = 1
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            ContextParallelBatchSplitUtils.create_context_parallel_split_uniform(
                mask,
                sp_group=dist.GroupMember.WORLD,
            )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen < 128 * world_size:
                start_offset, end_offset = 0, seqlen
            else:
                expected_seqlen_per_rank = math.ceil(seqlen / world_size)
                start_offset = expected_seqlen_per_rank * rank
                end_offset = min(expected_seqlen_per_rank * (rank + 1), seqlen)

            torch.testing.assert_close(data[:, start_offset:end_offset], split_data)

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    def test_split_batch_zigzag_singlebatch(self, seqlen: int, world_size: int):
        batch_size: int = 1
        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )
        mask = torch.full(
            (batch_size, seqlen), 1 << 1, dtype=torch.int64, device="cuda"
        )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            ContextParallelBatchSplitUtils.create_context_parallel_split_zigzag(
                mask,
                sp_group=dist.GroupMember.WORLD,
            )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen < 128 * world_size:
                expected_split_data = data
                expected_indices = np.arange(seqlen)
            else:
                num_chunks = world_size * 2
                base_size = seqlen // num_chunks
                remainder = seqlen % num_chunks
                chunk_sizes = [
                    base_size + 1 if i < remainder else base_size
                    for i in range(num_chunks)
                ]

                indices = np.arange(seqlen)
                chunks = []
                start = 0
                for size in chunk_sizes:
                    chunks.append(indices[start : start + size])
                    start += size

                # Each rank should get its corresponding chunk and the symmetric one.
                expected_first = chunks[rank]
                expected_second = chunks[-rank - 1]
                expected_indices = np.concatenate([expected_first, expected_second])

                expected_split_data = data[:, expected_indices, :, :]

            torch.testing.assert_close(expected_split_data, split_data)

    @pytest.mark.parametrize(
        "seqlen",
        [57, 256, 335, 684, 1024, 2003, 3712],
        ids=lambda x: f"seq={x}",
    )
    @pytest.mark.parametrize("world_size", [1, 4], ids=["world=1", "world=4"])
    @pytest.mark.parametrize("mask_type", ["causal", "full"])
    @pytest.mark.parametrize("mask_repr", ["bitfield", "full"])
    def test_split_batch_makespan_min_singlebatch(
        self, seqlen: int, world_size: int, mask_type: str, mask_repr: str
    ):
        batch_size = 1

        data = torch.randn(
            (batch_size, seqlen, self.num_heads, self.head_dim),
            device="cuda",
        )

        if mask_type == "causal":
            if mask_repr == "bitfield":
                mask = torch.full(
                    (batch_size, seqlen),
                    (1 << 62) | 1,
                    dtype=torch.int64,
                    device="cuda",
                )
            else:
                mask = torch.tril(
                    torch.ones(
                        (batch_size, seqlen, seqlen), dtype=torch.bool, device="cuda"
                    )
                )
        elif mask_type == "full":
            if mask_repr == "bitfield":
                mask = torch.full(
                    (batch_size, seqlen), (1 << 1), dtype=torch.int64, device="cuda"
                )
            else:
                mask = torch.ones(
                    (batch_size, seqlen, seqlen), dtype=torch.bool, device="cuda"
                )

        store = FakeStore()
        for rank in range(world_size):
            self.teardown_method()

            dist.init_process_group(
                "fake", rank=rank, world_size=world_size, store=store
            )

            if mask_repr == "bitfield":
                ContextParallelBatchSplitUtils.create_context_parallel_split_bitfield_makespan_minimization(
                    mask,
                    sp_group=dist.GroupMember.WORLD,
                )
            else:
                ContextParallelBatchSplitUtils.create_context_parallel_split_makespan_minimization(
                    mask,
                    sp_group=dist.GroupMember.WORLD,
                )

            split_data = ContextParallelBatchSplitUtils.split_batch(
                data,
                sp_group=dist.GroupMember.WORLD,
            )

            if world_size == 1 or seqlen <= 128 * 4:
                expected_split_data = data
                expected_indices = torch.arange(seqlen, device="cuda")
            else:
                num_blocks = math.ceil(seqlen / 128)

                if mask_type == "causal":
                    assignments = get_causal_assignments(num_blocks)
                elif mask_type == "full":
                    assignments = get_full_assignments(num_blocks)

                expected_indices = torch.nonzero(assignments == rank, as_tuple=True)[0]
                expected_indices = expected_indices[expected_indices < data.shape[1]]

                expected_split_data = data[:, expected_indices]

            torch.testing.assert_close(expected_split_data, split_data)


@instantiate_parametrized_tests
class TestBitfieldContextParallelismClass(GlooDistributedTestBase):

    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2, 4], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [64, 256, 336, 400, 1024], name_fn=lambda x: f"seq={x}")
    def test(self, batch_size: int, seq_len: int) -> tuple[torch.Tensor, ...]:
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.bfloat16
            ).normal_(mean=0, std=0.5),
        )
        key = key[:, :, :4]
        value = value[:, :, :4]

        for t in [query, key, value]:
            t.requires_grad_()

        # mask = torch.full(
        #     (batch_size, seq_len), 1 << 1, dtype=torch.int64, device="cuda"
        # )
        mask = torch.full(
            (batch_size, seq_len),
            (1 << 62) | 1 | (1 << 1) | (1 << 2),
            dtype=torch.int64,
            device="cuda",
        )
        mask[:, 32:44] = 1 << 1
        # if seq_len >= 256:
        #     mask[:, 180:280] = 1 << 2

        ref_out: torch.Tensor = bitfield_attn_func(
            query, key, value, bitfield_mask=mask
        )

        seq_len = query.size(1)
        base_chunk = (seq_len // self.world_size // 128) * 128
        chunk_sizes = [base_chunk] * (self.world_size - 1)
        chunk_sizes.append(seq_len - sum(chunk_sizes))

        local_query = (
            torch.split(query, chunk_sizes, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_key = (
            torch.split(key, chunk_sizes, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )
        local_value = (
            torch.split(value, chunk_sizes, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        assert (
            local_key.shape
            == local_value.shape
            == (batch_size, chunk_sizes[self.rank], 4, 64)
        )

        offsets_per_rank = torch.split(
            torch.arange(query.shape[1], device="cuda"), chunk_sizes, dim=0
        )
        ContextParallelBatchSplitUtils.set_context_parallel_offsets_cache(
            offsets_per_rank
        )
        compressed_mask = ContextParallelBatchSplitUtils.get_local_compressed_mask(
            mask, dist.GroupMember.WORLD
        )

        cp_out: torch.Tensor = ContextParallelBitfieldAttention.apply(
            local_query,
            local_key,
            local_value,
            mask,
            compressed_mask,
            offsets_per_rank,
            dist.GroupMember.WORLD,
        )

        torch.testing.assert_close(
            torch.split(ref_out, chunk_sizes, dim=1)[self.rank].contiguous(),
            cp_out,
            rtol=5e-3,
            atol=5e-3,
        )

        # ========================================================================
        # Check backward
        # ========================================================================

        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, [query, key, value], dout)

        cp_dout = torch.split(dout, chunk_sizes, dim=1)[self.rank].contiguous()
        cp_dq, cp_dk, cp_dv = torch.autograd.grad(
            cp_out, [local_query, local_key, local_value], cp_dout
        )

        torch.testing.assert_close(
            torch.split(ref_dq, chunk_sizes, dim=1)[self.rank].contiguous(),
            cp_dq,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.split(ref_dk, chunk_sizes, dim=1)[self.rank].contiguous(),
            cp_dk,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.split(ref_dv, chunk_sizes, dim=1)[self.rank].contiguous(),
            cp_dv,
            rtol=5e-3,
            atol=5e-3,
        )


@instantiate_parametrized_tests
class TestFlashAttentionContextParallelismClass(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2, 4], name_fn=lambda x: f"bs={x}")
    @parametrize("seq_len", [64, 256, 336, 400, 1024], name_fn=lambda x: f"seq={x}")
    def test(self, batch_size: int, seq_len: int):
        query, key, value = torch.unbind(
            torch.randn(
                (3, batch_size, seq_len, 8, 64), device="cuda", dtype=torch.bfloat16
            ).normal_(mean=0, std=0.5),
        )

        for t in [query, key, value]:
            t.requires_grad_()

        ref_out: torch.Tensor = sdpa(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=False,
        ).transpose(1, 2)

        local_query = (
            torch.chunk(query, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        local_key = (
            torch.chunk(key, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        local_value = (
            torch.chunk(value, self.world_size, dim=1)[self.rank]
            .requires_grad_()
            .contiguous()
        )

        assert (
            local_key.shape
            == local_value.shape
            == (batch_size, seq_len // self.world_size, 8, 64)
        )

        cp_out: torch.Tensor = ContextParallelFlashAttention.apply(
            local_query,
            local_key,
            local_value,
            dist.GroupMember.WORLD,
        )

        torch.testing.assert_close(
            torch.chunk(ref_out, self.world_size, dim=1)[self.rank],
            cp_out,
            rtol=5e-3,
            atol=5e-3,
        )

        # ========================================================================
        # Check backward
        # ========================================================================

        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(ref_out, [query, key, value], dout)

        cp_dout = torch.chunk(dout, self.world_size, dim=1)[self.rank].contiguous()
        cp_dq, cp_dk, cp_dv = torch.autograd.grad(
            cp_out, [local_query, local_key, local_value], cp_dout
        )

        torch.testing.assert_close(
            torch.chunk(ref_dq, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dq,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dk, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dk,
            rtol=5e-3,
            atol=5e-3,
        )
        torch.testing.assert_close(
            torch.chunk(ref_dv, self.world_size, dim=1)[self.rank].contiguous(),
            cp_dv,
            rtol=5e-3,
            atol=5e-3,
        )


@instantiate_parametrized_tests
class TestVarlenRingAttentionClass(GlooDistributedTestBase):
    """
    Tests ContextParallelVarlenRingAttention with zigzag partitioning.

    Covers both perfectly-divisible and non-divisible sequence lengths, with
    and without padding tokens, to validate:
      - correct offset-aware causal masking across ranks
      - padded all_gather for unequal seqlen_per_rank
      - correct inv_perm construction
      - backward gradients
    """

    @property
    def world_size(self) -> int:
        return 2

    @parametrize("batch_size", [1, 2], name_fn=lambda x: f"bs={x}")
    @parametrize(
        "seq_len",
        [
            64,   # divisible by 2*P=4
            66,   # 66 % 4 = 2 (non-divisible)
            67,   # 67 % 4 = 3 (non-divisible)
            128,  # divisible
            130,  # 130 % 4 = 2 (non-divisible)
        ],
        name_fn=lambda x: f"seq={x}",
    )
    @parametrize(
        "with_padding",
        [False, True],
        name_fn=lambda x: "padded" if x else "nopad",
    )
    def test(self, batch_size: int, seq_len: int, with_padding: bool) -> None:
        nheads  = 4
        headdim = 32
        sp_size = self.world_size   # 2
        sp_rank = self.rank
        device  = torch.device("cuda")
        dtype   = torch.bfloat16

        # ── Build full Q/K/V with the same random seed on all ranks ─────────
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, nheads, headdim,
                        device=device, dtype=dtype).normal_(mean=0, std=0.5)
        k = torch.randn(batch_size, seq_len, nheads, headdim,
                        device=device, dtype=dtype).normal_(mean=0, std=0.5)
        v = torch.randn(batch_size, seq_len, nheads, headdim,
                        device=device, dtype=dtype).normal_(mean=0, std=0.5)
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()

        # ── Build attention mask ─────────────────────────────────────────────
        # mask: (batch, seq_len) with 1=valid, 0=padding
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        if with_padding and batch_size > 1:
            # Make the second sample shorter by padding its last quarter
            mask[1, seq_len * 3 // 4:] = False

        # ── Reference: causal SDPA on full (unsharded) sequence ─────────────
        # Build explicit lower-triangular mask per sample to handle padding.
        # Shape: (batch, 1, seq_len, seq_len)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
        # Mask out padding K tokens
        pad_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        full_mask = causal_mask & pad_mask
        # Also mask out padding Q tokens from contributing
        q_pad_mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len, 1)
        attn_mask_float = torch.where(
            full_mask,
            torch.zeros(1, device=device, dtype=torch.float32),
            torch.full((1,), float("-inf"), device=device, dtype=torch.float32),
        )
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(),   # (batch, nheads, seq, D)
            k.transpose(1, 2).float(),
            v.transpose(1, 2).float(),
            attn_mask=attn_mask_float,
        ).transpose(1, 2).to(dtype)   # (batch, seq, nheads, D)
        # Zero out padded-Q positions in the reference output
        ref_out = ref_out * mask.unsqueeze(-1).unsqueeze(-1)

        # ── Zigzag partition: rank i gets chunks i and 2P-1-i ───────────────
        local_idx = _zigzag_local_indices(sp_rank, sp_size, seq_len, device)
        local_len = local_idx.shape[0]

        # Chunk boundary (may differ by 1 for non-divisible seq_len)
        _total_chunks = 2 * sp_size
        _base   = seq_len // _total_chunks
        _extra  = seq_len %  _total_chunks
        chunk_a_size = _base + (1 if sp_rank < _extra else 0)
        chunk_b_size = local_len - chunk_a_size

        lo_a = int(local_idx[0].item())
        lo_b = int(local_idx[chunk_a_size].item())

        # ── Pack chunk_a and chunk_b separately (2B sub-sequences) ──────────
        local_q = q[:, local_idx]   # (batch, local_len, nheads, D)
        local_k = k[:, local_idx]
        local_v = v[:, local_idx]

        q_a = local_q[:, :chunk_a_size]   # (batch, chunk_a_size, nheads, D)
        q_b = local_q[:, chunk_a_size:]
        k_a = local_k[:, :chunk_a_size]
        k_b = local_k[:, chunk_a_size:]
        v_a = local_v[:, :chunk_a_size]
        v_b = local_v[:, chunk_a_size:]

        mask_a = mask[:, local_idx[:chunk_a_size]]   # (batch, chunk_a_size)
        mask_b = mask[:, local_idx[chunk_a_size:]]   # (batch, chunk_b_size)

        idx_a, cu_seqlens_a, max_seqlen_a = _get_unpad_data(mask_a.int())
        idx_b, cu_seqlens_b, max_seqlen_b = _get_unpad_data(mask_b.int())

        def _pack(x_chunk: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            """x_chunk: (batch, chunk_sz, nheads, D) → (total_valid, nheads, D)"""
            B, C, H, D = x_chunk.shape
            return index_first_axis(x_chunk.reshape(B * C, H, D), idx)

        pq = torch.cat([_pack(q_a, idx_a), _pack(q_b, idx_b)], dim=0)
        pk = torch.cat([_pack(k_a, idx_a), _pack(k_b, idx_b)], dim=0)
        pv = torch.cat([_pack(v_a, idx_a), _pack(v_b, idx_b)], dim=0)
        pq = pq.detach().requires_grad_()
        pk = pk.detach().requires_grad_()
        pv = pv.detach().requires_grad_()

        # ── Metadata for ring attention ──────────────────────────────────────
        cu_seqlens_q = torch.cat(
            [cu_seqlens_a, cu_seqlens_a[-1] + cu_seqlens_b[1:]]
        )   # (2B+1,) int32
        max_seqlen_q = max(int(max_seqlen_a), int(max_seqlen_b))

        valid_lens   = mask.sum(dim=1).long()           # (batch,)
        q_offsets_a  = valid_lens.clamp(max=lo_a).int()
        q_offsets_b  = valid_lens.clamp(max=lo_b).int()
        q_seq_offsets = torch.cat([q_offsets_a, q_offsets_b])   # (2B,) int32

        k_lens = torch.cat([valid_lens, valid_lens])   # (2B,)
        cu_seqlens_k_global = torch.zeros(
            2 * batch_size + 1, dtype=torch.int32, device=device
        )
        cu_seqlens_k_global[1:] = k_lens.cumsum(0).int()

        max_seqlen_k = int(valid_lens.max().item())

        # ── Ring attention forward ───────────────────────────────────────────
        cp_out = ContextParallelVarlenRingAttention.apply(
            pq, pk, pv,
            dist.GroupMember.WORLD,
            cu_seqlens_q, cu_seqlens_k_global, q_seq_offsets,
            max_seqlen_q, max_seqlen_k,
        )   # (total_local_q, nheads, D)

        # ── Unpack cp_out back to (batch, local_len, nheads, D) ─────────────
        total_a = int(cu_seqlens_a[-1].item())
        total_b = int(cu_seqlens_b[-1].item())
        out_a_packed = cp_out[:total_a]   # (total_valid_a, nheads, D)
        out_b_packed = cp_out[total_a:]   # (total_valid_b, nheads, D)

        cp_out_full = torch.zeros(
            batch_size, local_len, nheads, headdim, device=device, dtype=dtype
        )
        # scatter chunk_a tokens back
        for b in range(batch_size):
            a_start = int(cu_seqlens_a[b].item())
            a_end   = int(cu_seqlens_a[b + 1].item())
            b_start = int(cu_seqlens_b[b].item())
            b_end   = int(cu_seqlens_b[b + 1].item())
            # Valid positions within chunk_a/b for sample b
            valid_a = torch.where(mask_a[b])[0]
            valid_b = torch.where(mask_b[b])[0]
            cp_out_full[b, valid_a, :, :] = out_a_packed[a_start:a_end]
            cp_out_full[b, chunk_a_size + valid_b, :, :] = out_b_packed[b_start:b_end]

        # ── Forward comparison ───────────────────────────────────────────────
        ref_local = ref_out[:, local_idx]   # (batch, local_len, nheads, D)
        torch.testing.assert_close(
            cp_out_full,
            ref_local,
            atol=5e-3,
            rtol=5e-3,
            msg=(
                f"Forward mismatch: rank={sp_rank} seq={seq_len} "
                f"bs={batch_size} padded={with_padding}"
            ),
        )

        # ── Backward comparison ──────────────────────────────────────────────
        dout = torch.randn_like(ref_out).normal_(mean=0, std=0.5)
        # Reference gradients (only for local Q/K/V positions)
        ref_dq, ref_dk, ref_dv = torch.autograd.grad(
            ref_out, [q, k, v], dout
        )

        # CP gradients
        cp_dout_packed = torch.cat(
            [
                _pack(dout[:, local_idx[:chunk_a_size]], idx_a),
                _pack(dout[:, local_idx[chunk_a_size:]], idx_b),
            ],
            dim=0,
        )
        cp_dq_packed, cp_dk_packed, cp_dv_packed = torch.autograd.grad(
            cp_out, [pq, pk, pv], cp_dout_packed
        )

        def _unpack_grad(
            grad_packed: torch.Tensor,
            cu_a: torch.Tensor,
            cu_b: torch.Tensor,
            mask_a_b: torch.Tensor,
            mask_b_b: torch.Tensor,
        ) -> torch.Tensor:
            """Scatter packed gradient back to (batch, local_len, nheads, D)."""
            tot_a = int(cu_a[-1].item())
            out = torch.zeros(batch_size, local_len, nheads, headdim,
                              device=device, dtype=dtype)
            for b in range(batch_size):
                a_s, a_e = int(cu_a[b].item()), int(cu_a[b + 1].item())
                b_s, b_e = int(cu_b[b].item()), int(cu_b[b + 1].item())
                valid_a = torch.where(mask_a_b[b])[0]
                valid_b = torch.where(mask_b_b[b])[0]
                out[b, valid_a] = grad_packed[a_s:a_e].to(dtype)
                out[b, chunk_a_size + valid_b] = grad_packed[tot_a + b_s: tot_a + b_e].to(dtype)
            return out

        cp_dq = _unpack_grad(cp_dq_packed, cu_seqlens_a, cu_seqlens_b, mask_a, mask_b)
        cp_dk = _unpack_grad(cp_dk_packed, cu_seqlens_a, cu_seqlens_b, mask_a, mask_b)
        cp_dv = _unpack_grad(cp_dv_packed, cu_seqlens_a, cu_seqlens_b, mask_a, mask_b)

        torch.testing.assert_close(
            cp_dq,
            ref_dq[:, local_idx].to(dtype),
            atol=5e-3,
            rtol=5e-3,
            msg=f"dQ mismatch: rank={sp_rank} seq={seq_len}",
        )
        torch.testing.assert_close(
            cp_dk,
            ref_dk[:, local_idx].to(dtype),
            atol=5e-3,
            rtol=5e-3,
            msg=f"dK mismatch: rank={sp_rank} seq={seq_len}",
        )
        torch.testing.assert_close(
            cp_dv,
            ref_dv[:, local_idx].to(dtype),
            atol=5e-3,
            rtol=5e-3,
            msg=f"dV mismatch: rank={sp_rank} seq={seq_len}",
        )
