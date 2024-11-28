import csv
import heapq
import time
from collections import defaultdict
from enum import Enum
from types import MethodType
from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    apply_rotary_pos_emb,
)

from cornstarch.testing.utils import get_random_mask


class DistributeMethod(Enum):
    random = "Random"
    makespan_min = "Makespan Min"
    naive_ring = "Naive Ring"
    stripe = "Striped"


class AttentionMaskType(Enum):
    causal_blockwise = "causal_blockwise"
    prefix_lm_causal = "prefix_lm_causal"
    prefix_lm_document = "prefix_lm_document"
    encoder_embedded = "encoder_embedded"
    multimodal_packed = "multimodal_packed"


def get_naive_ring_query_positions(
    attention_mask: torch.BoolTensor, seq_len: int, world_size: int
) -> torch.Tensor:
    num_tokens_per_rank = seq_len // world_size
    return torch.arange(world_size, device="cuda").repeat_interleave(
        num_tokens_per_rank
    )


def get_stripe_query_positions(
    attention_mask: torch.BoolTensor,
    seq_len: int,
    world_size: int,
    block_size: int = 2048,
) -> torch.Tensor:
    while seq_len < block_size * 2 * world_size:
        block_size //= 2

    assert seq_len % (block_size * 2) == 0
    num_blocks = seq_len // block_size

    half_num_blocks = num_blocks // 2

    block_indices_first_half = torch.arange(world_size, device="cuda").repeat(
        half_num_blocks // world_size
    )
    block_indices_second_half = (
        torch.arange(world_size, device="cuda")
        .repeat(half_num_blocks // world_size)
        .flip(0)
    )

    block_indices = torch.cat((block_indices_first_half, block_indices_second_half))
    assignments = block_indices.repeat_interleave(block_size)
    assert assignments.size(0) == seq_len, f"{assignments.size(0)} != {seq_len}"

    return assignments


random_cache = None


def get_random_query_positions(
    attention_mask: torch.BoolTensor, seq_len: int, world_size: int
) -> torch.Tensor:
    global random_cache
    if random_cache is not None:
        return random_cache

    block_size = 128
    assert seq_len % block_size == 0
    num_blocks = seq_len // block_size

    def generate_coprime_a(p, mod):
        while True:
            a = np.random.randint(1, p)
            if np.gcd(a, mod) == 1:
                return a

    # Hash function parameters
    p = 2**31  # Modulus
    a = generate_coprime_a(p, world_size)  # multiplier
    b = np.random.randint(0, p)  # increment
    offset = np.random.randint(0, p)

    token_indices = np.arange(num_blocks, dtype=np.int64)
    # Compute the hash for each index
    hash_values = (a * ((token_indices + offset) % p) + b) % p  # shape: [seq_len]

    # Determine assignment based on hash modulo 'mod'
    assignments = hash_values % world_size
    assignments = torch.as_tensor(assignments, dtype=torch.long, device="cuda")
    assignments = assignments.repeat_interleave(128)

    random_cache = assignments

    return assignments


makespan_cache = None


def get_makespan_min_query_positions(
    attention_mask: torch.BoolTensor, seq_len: int, world_size: int
) -> torch.Tensor:
    global makespan_cache
    if makespan_cache is not None:
        return makespan_cache

    block_size = 128
    assert seq_len % block_size == 0
    num_blocks = seq_len // block_size

    # compress the attention mask, for each 128x128 block, set True if any True in the block.
    block_mask = attention_mask.view(num_blocks, block_size, num_blocks, block_size)
    block_mask = torch.any(block_mask, dim=(1, 3))

    per_block_workloads_sorted, indices = torch.sort(
        block_mask.sum(dim=1), descending=True
    )

    loads_per_gpu = []
    for i in range(world_size):
        heapq.heappush(loads_per_gpu, (0, i, []))

    for indice, block_workload in zip(indices, per_block_workloads_sorted):
        load, gpu, block_indices = heapq.heappop(loads_per_gpu)
        block_indices.append(indice.item())
        heapq.heappush(
            loads_per_gpu, (load + block_workload.item(), gpu, block_indices)
        )

    assignments = torch.empty(num_blocks, dtype=torch.long, device="cuda")
    for _, gpu, block_indices in loads_per_gpu:
        for block_idx in block_indices:
            assignments[block_idx] = gpu
    assignments = assignments.repeat_interleave(128)

    makespan_cache = assignments

    return assignments


get_query_positions_dict = {
    DistributeMethod.random: get_random_query_positions,
    DistributeMethod.makespan_min: get_makespan_min_query_positions,
    DistributeMethod.naive_ring: get_naive_ring_query_positions,
    DistributeMethod.stripe: get_stripe_query_positions,
}


def get_mask_mode_based_on_query_positions(
    attention_mask: torch.Tensor,
    query_positions: torch.Tensor,
    rank: int,
):
    q_positions = torch.arange(query_positions.size(0), device="cuda")[
        query_positions == rank
    ]

    def mask_mod(b, h, q_idx, kv_idx):
        converted_q_idx = q_positions[q_idx]
        return attention_mask[converted_q_idx, kv_idx]

    return mask_mod


def create_layer():
    config: LlamaConfig = LlamaConfig.from_pretrained(
        "meta-llama/Meta-Llama-3-70B-Instruct"
    )
    config.num_attention_heads //= 8
    config.num_key_value_heads //= 8

    with torch.device("meta"):
        layer = LlamaAttention(config, layer_idx=0).to(dtype=torch.bfloat16)
    layer.to_empty(device="cuda")

    def forward(
        self: LlamaAttention,
        hidden_states: torch.Tensor,
        rank: int,
        world_size: int,
        query_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        original_seq_len: int,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # This simulates kv to be gathered from all GPUs
        key_states = torch.randn(
            (bsz, self.num_heads, original_seq_len, self.head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        value_states = torch.randn(
            (bsz, self.num_heads, original_seq_len, self.head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )

        block_mask = create_block_mask(
            get_mask_mode_based_on_query_positions(
                attention_mask, query_positions, rank
            ),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=original_seq_len,
            device="cuda",
            _compile=True,
        )

        attn_output = torch.compile(flex_attention, backend="inductor", fullgraph=True)(
            query_states,
            key_states,
            value_states,
            block_mask=block_mask,
            enable_gqa=True,
            return_lse=False,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        return attn_output, None, None

    layer.forward = MethodType(forward, layer)

    return layer


@torch.no_grad()
def run_profile(
    seq_len: int,
    world_size: int,
    attention_mask_type: AttentionMaskType,
):
    num_batch = 1
    num_iterations = 50

    # Create a layer
    layer: LlamaAttention = create_layer()

    # value: rank to list of times
    times: dict[DistributeMethod, list[list[float]]] = defaultdict(list)

    print(
        f"Running attention {attention_mask_type.value} with seq_len={seq_len}, world_size={world_size}"
    )
    for iter in range(num_iterations):

        # Create attention mask.
        # This attention mask is used for every distribution method to compare their perofrmance.
        attention_mask: torch.BoolTensor = get_random_mask(
            seq_len, attention_mask_type.value
        ).to(dtype=torch.bool, device="cuda")

        # Reset caches
        global makespan_cache
        global random_cache
        makespan_cache = None
        random_cache = None

        start_events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(world_size)]
            for _ in range(len(DistributeMethod))
        ]
        end_events = [
            [torch.cuda.Event(enable_timing=True) for _ in range(world_size)]
            for _ in range(len(DistributeMethod))
        ]

        for i, distribute_method in enumerate(DistributeMethod):
            # Get query position for this rank
            query_positions = get_query_positions_dict[distribute_method](
                attention_mask, seq_len, world_size
            )

            for rank in range(world_size):
                # Create fake tensors
                num_tokens_for_rank = (query_positions == rank).sum().item()
                hidden_states = torch.randn(
                    (num_batch, num_tokens_for_rank, layer.config.hidden_size),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                position_ids = (
                    torch.arange(num_tokens_for_rank, device="cuda")
                    .unsqueeze(0)
                    .expand(num_batch, -1)
                )

                # Warm up run to compile
                layer(
                    hidden_states,
                    position_ids=position_ids,
                    rank=rank,
                    world_size=world_size,
                    attention_mask=attention_mask,
                    query_positions=query_positions,
                    original_seq_len=seq_len,
                )

                start_events[i][rank].record()
                layer(
                    hidden_states,
                    position_ids=position_ids,
                    rank=rank,
                    world_size=world_size,
                    attention_mask=attention_mask,
                    query_positions=query_positions,
                    original_seq_len=seq_len,
                )
                end_events[i][rank].record()

        torch.cuda.synchronize()

        for i, distribute_method in enumerate(DistributeMethod):
            for rank in range(world_size):
                elapsed_time_ms = start_events[i][rank].elapsed_time(
                    end_events[i][rank]
                )

                if not times[distribute_method]:
                    times[distribute_method] = [[] for _ in range(world_size)]

                times[distribute_method][rank].append(elapsed_time_ms)

    with open("profile_context_parallel_result.csv", "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attention_mask_type",
                "seq_len",
                "world_size",
                "distribute_method",
                "per_rank_time (ms)",
                "max_time (ms)",
                "average_time (ms)",
                "coefficient_variation",
            ],
        )
        if f.tell() == 0:
            writer.writeheader()

        random_sample_per_rank_time = np.random.randint(num_iterations)

        for distribute_method in DistributeMethod:

            dist_times = np.array(times[distribute_method])
            average_per_trial = np.mean(dist_times, axis=0)
            max_per_trial = np.max(dist_times, axis=0)
            std_per_trial = np.std(dist_times, axis=0)
            cv = std_per_trial / average_per_trial

            writer.writerow(
                {
                    "attention_mask_type": attention_mask_type.value,
                    "seq_len": seq_len,
                    "world_size": world_size,
                    "distribute_method": distribute_method.value,
                    "per_rank_time (ms)": dist_times[
                        :, random_sample_per_rank_time
                    ].tolist(),
                    "max_time (ms)": np.mean(max_per_trial),
                    "average_time (ms)": np.mean(average_per_trial),
                    "coefficient_variation": np.mean(cv),
                }
            )


if __name__ == "__main__":
    import os

    index = int(os.environ["CUDA_VISIBLE_DEVICES"])
    type = [
        AttentionMaskType.prefix_lm_causal,
        AttentionMaskType.prefix_lm_document,
        AttentionMaskType.encoder_embedded,
        AttentionMaskType.multimodal_packed,
    ]

    for seq_len in [16384, 16384 * 2, 16384 * 4]:
        # for seq_len in [16384 * 4, 16384 * 8]:
        # for num_rank in [8, 16]:
        for num_rank in [8]:
            seed = time.time()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(int(seed))
            run_profile(seq_len, num_rank, type[index])
