import os
import torch
import torch.distributed as dist
from torch.optim import Adam
from transformers import LlamaConfig, LlamaForCausalLM
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.shardformer.policies.auto_policy import _fullname
from cornstarch.shardformer.policies.auto_policy import get_autopolicy
import random
import numpy as np

def reset_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

def setup_distributed(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def data_gen_fn(num_batches, seq_len, vocab_size):
    input_ids = torch.randint(0, 2048, (num_batches, seq_len))
    attn_mask_rand = torch.randint(0, 2, (num_batches, seq_len, seq_len)) # B, L, L
    # attn_mask_rand[0, 0, 0] = 0
    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask_rand,
        "labels": input_ids,
    }

def train(rank, world_size, tp_size, pp_size, sp_size, sp_mode):
    setup_distributed(rank, world_size)
    reset_seed()

    # Model configuration
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=4,
        use_cache=False,
        _attn_implementation="eager",
    )

    # Create model and move to GPU
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    optimizer = Adam(model.parameters(), lr=1e-3)

    # HybridParallel configuration
    plugin_config = dict(
        tp_size=tp_size,
        pp_size=pp_size,
        sp_size=sp_size,
        enable_sequence_parallelism=sp_size > 1,
        sequence_parallelism_mode=sp_mode,
        precision="bf16",
    )

    # Create plugin and booster
    policy = get_autopolicy(_fullname(model))
    plugin = HybridParallelPlugin(**plugin_config, custom_policy=policy)
    booster = Booster(plugin=plugin)

    # Boost model and optimizer
    model, optimizer, _, _, _ = booster.boost(model, optimizer)

    # Training loop
    num_epochs = 1
    num_batches = 1
    seq_len = 512
    vocab_size = 32000

    for epoch in range(num_epochs):
        for step in range(num_batches):
            # Generate dummy data
            data = data_gen_fn(1, seq_len, vocab_size)
            inputs = {k: v.to("cuda") for k, v in data.items()}

            # Forward pass
            outputs = model(**inputs)
            # print(f"outputs: {outputs}")
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            booster.backward(loss, optimizer)
            optimizer.step()

            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{num_batches}, Loss: {loss.item()}")

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--sp_size", type=int, default=4)
    parser.add_argument("--sp_mode", type=str, default="ring_attn", choices=["ring_attn", "ring_attn_zig_zag", "ring_attn_uniform", "ring_attn_optimal"])
    return parser.parse_args()

"""
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_llm_ring.py --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_llm_ring.py --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn_zig_zag
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_llm_ring.py --world_size 4 --tp_size 1 --pp_size 1 --sp_size 4 --sp_mode ring_attn_optimal

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_llm_ring.py --world_size 4 --tp_size 2 --pp_size 1 --sp_size 2 --sp_mode ring_attn
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_llm_ring.py --world_size 4 --tp_size 4 --pp_size 1 --sp_size 1
"""

if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size
    tp_size = args.tp_size
    pp_size = args.pp_size
    sp_size = args.sp_size
    sp_mode = args.sp_mode

    assert world_size == tp_size * pp_size * sp_size

    import torch.multiprocessing as mp
    mp.spawn(train, args=(world_size, tp_size, pp_size, sp_size, sp_mode), nprocs=world_size)