"""Benchmark FLA Gated DeltaNet summaries under run-aware context parallelism.

Run on one GPU per rank, for example::

    torchrun --nproc-per-node=2 --module \
        examples.distributed.benchmark_gated_delta_cp \
        --splitter head-tail --sequence-length 8192 --iterations 20

For multiple nodes, add torchrun's rendezvous arguments. Each process binds
``LOCAL_RANK`` to its node-local CUDA device before NCCL initialization.

The report includes recurrent-summary bytes, collective time, peak CUDA
memory, and forward/backward token throughput versus stock FLA contiguous CP
at the same global sequence length. It intentionally benchmarks the recurrent
operator; the convolution halo is independent and bounded by kernel size.
"""
from __future__ import annotations

import argparse
import functools
import time

import torch
import torch.distributed as dist

from .common import init_distributed

from cornstarch.distributed.context_parallel.gated_delta import (
    build_gated_delta_metadata,
)
from cornstarch.distributed.context_parallel.gated_delta_fla import (
    CornstarchRunAwareFLACPContext,
    install_run_aware_fla_dispatch,
)
from cornstarch.distributed.context_parallel.splitters import (
    HeadTailContextParallelSplitter,
    UniformContextParallelSplitter,
)


def _time_step(fn, iterations: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--splitter", choices=("uniform", "head-tail"), default="head-tail")
    args = parser.parse_args()

    rank, world_size, device = init_distributed()
    if args.sequence_length % (2 * world_size):
        raise SystemExit("sequence length must be divisible by 2 * world size")
    install_run_aware_fla_dispatch()
    from fla.ops.cp import build_cp_context
    from fla.ops.cp import chunk_delta_h as cp_delta_h
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    mask = torch.ones(1, args.sequence_length, dtype=torch.bool)
    splitter = (
        HeadTailContextParallelSplitter()
        if args.splitter == "head-tail"
        else UniformContextParallelSplitter()
    )
    offsets = splitter.offsets_for_size(mask, world_size)
    metadata = build_gated_delta_metadata(offsets, mask)
    local_runs = metadata.local_runs(rank)
    lengths = [run.length for run in local_runs]
    cu_cpu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    context = CornstarchRunAwareFLACPContext(
        group=dist.group.WORLD,
        cu_seqlens=cu_cpu.to(device),
        cu_seqlens_cpu=cu_cpu,
        metadata=metadata,
        local_run_indices=tuple(run.index for run in local_runs),
    )

    local_length = len(offsets[rank])

    def local_inputs(seed: int) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        q = torch.randn(
            1, local_length, args.heads, args.key_dim,
            generator=generator, device=device, dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            1, local_length, args.heads, args.value_dim,
            generator=generator, device=device, dtype=torch.bfloat16,
        )
        g = -torch.rand(
            1, local_length, args.heads, generator=generator, device=device
        )
        beta = torch.rand(
            1, local_length, args.heads, generator=generator, device=device
        ).sigmoid()
        return q, k, v, g, beta

    custom_inputs = local_inputs(1234)
    custom_collective_seconds = 0.0
    custom_collective_bytes = 0
    custom_collectives = 0
    original_all_reduce = dist.all_reduce

    def measured_all_reduce(tensor, *call_args, **call_kwargs):
        nonlocal custom_collective_seconds, custom_collective_bytes
        nonlocal custom_collectives
        is_summary = tensor.ndim == 4 and tensor.shape[0] == len(metadata.runs)
        if not is_summary:
            return original_all_reduce(tensor, *call_args, **call_kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_all_reduce(tensor, *call_args, **call_kwargs)
        torch.cuda.synchronize()
        custom_collective_seconds += time.perf_counter() - started
        custom_collective_bytes += tensor.numel() * tensor.element_size()
        custom_collectives += 1
        return result

    def step(inputs, cp_context=None):
        tensors = [item.detach().requires_grad_(True) for item in inputs]
        output, _ = chunk_gated_delta_rule(
            *tensors,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cp_context=cp_context,
        )
        output.float().square().mean().backward()

    torch.cuda.reset_peak_memory_stats()
    dist.all_reduce = measured_all_reduce
    try:
        custom_seconds = _time_step(
            functools.partial(step, custom_inputs, context), args.iterations
        )
    finally:
        dist.all_reduce = original_all_reduce
    custom_peak = torch.cuda.max_memory_allocated()

    del custom_inputs
    torch.cuda.empty_cache()
    contiguous_inputs = local_inputs(5678)
    global_cu = torch.tensor(
        [0, args.sequence_length], dtype=torch.int32, device=device
    )
    contiguous_context = build_cp_context(global_cu, dist.group.WORLD)
    baseline_collective_seconds = 0.0
    baseline_collective_bytes = 0
    baseline_collectives = 0
    original_gather = cp_delta_h.all_gather_into_tensor

    def measured_gather(tensor, *call_args, **call_kwargs):
        nonlocal baseline_collective_seconds, baseline_collective_bytes
        nonlocal baseline_collectives
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = original_gather(tensor, *call_args, **call_kwargs)
        torch.cuda.synchronize()
        baseline_collective_seconds += time.perf_counter() - started
        baseline_collective_bytes += (
            tensor.numel() * tensor.element_size() * world_size
        )
        baseline_collectives += 1
        return result

    torch.cuda.reset_peak_memory_stats()
    cp_delta_h.all_gather_into_tensor = measured_gather
    try:
        contiguous_seconds = _time_step(
            functools.partial(step, contiguous_inputs, contiguous_context),
            args.iterations,
        )
    finally:
        cp_delta_h.all_gather_into_tensor = original_gather
    contiguous_peak = torch.cuda.max_memory_allocated()
    measured_steps = args.iterations + 3
    report = {
        "splitter": args.splitter,
        "world_size": world_size,
        "global_runs": len(metadata.runs),
        "custom_summary_bytes_per_step_per_rank": (
            custom_collective_bytes // measured_steps
        ),
        "custom_collective_ms_per_step": (
            1000 * custom_collective_seconds / measured_steps
        ),
        "contiguous_summary_bytes_per_step_per_rank": (
            baseline_collective_bytes // measured_steps
        ),
        "contiguous_collective_ms_per_step": (
            1000 * baseline_collective_seconds / measured_steps
        ),
        "custom_tokens_per_second": args.sequence_length / custom_seconds,
        "contiguous_tokens_per_second": args.sequence_length / contiguous_seconds,
        "throughput_ratio_custom_over_contiguous": (
            contiguous_seconds / custom_seconds
        ),
        "custom_peak_bytes": custom_peak,
        "contiguous_peak_bytes": contiguous_peak,
        "custom_collectives_per_step": custom_collectives / measured_steps,
        "contiguous_collectives_per_step": baseline_collectives / measured_steps,
    }
    if rank == 0:
        print(report)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
