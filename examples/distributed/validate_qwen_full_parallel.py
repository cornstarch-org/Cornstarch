"""Run the release-gate Qwen parallelism topologies on real GPUs.

This is intentionally an executable acceptance test instead of a unit-test
fixture: the MoE cases need 32 ranks (2-way DP x PP x CP x TP x EP) and the
dense case needs 16 ranks (2-way DP x PP x CP x TP).  Launch one process per
GPU with ``torchrun``.  For example, on a single 32-GPU node::

    torchrun --nproc-per-node=32 --module \
        examples.distributed.validate_qwen_full_parallel \
        --case moe-hybrid-5d

The test drives the public ``ParallelizationPlan`` surface, including its DP
sampler, CP splitter, 1F1B schedule, CP/DP gradient synchronization, and an
optimizer update.  Success therefore means every configured mesh axis took
part in a finite forward/backward training step; it is not a construction-only
smoke test.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from importlib import metadata as importlib_metadata

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from cornstarch.distributed import (
    HeadTailContextParallelSplitter,
    ParallelConfig,
    ParallelizationPlan,
    UniformContextParallelSplitter,
)
from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture, from_hf_config


VOCAB_SIZE = 256


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    moe: bool
    layer_types: tuple[str, str]
    dp: int = 2
    pp: int = 2
    cp: int = 2
    tp: int = 2
    ep: int = 1

    @property
    def world_size(self) -> int:
        return self.dp * self.pp * self.cp * self.tp * self.ep


CASES = {
    "moe-hybrid-5d": AcceptanceCase(
        "moe-hybrid-5d", True, ("full_attention", "linear_attention"), ep=2
    ),
    "moe-attention-5d": AcceptanceCase(
        "moe-attention-5d", True, ("full_attention", "full_attention"), ep=2
    ),
    "dense-hybrid-4d": AcceptanceCase(
        "dense-hybrid-4d", False, ("full_attention", "linear_attention")
    ),
}


class _TokenDataset(Dataset):
    def __init__(self, length: int, sequence_length: int) -> None:
        self.length = length
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(10_000 + index)
        input_ids = torch.randint(
            0, VOCAB_SIZE, (self.sequence_length,), generator=generator
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def _config(case: AcceptanceCase):
    common = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=list(case.layer_types),
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        tie_word_embeddings=False,
    )
    if case.moe:
        return Qwen3_5MoeTextConfig(
            **common,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            output_router_logits=True,
        )
    return Qwen3_5TextConfig(**common, intermediate_size=128)


def _collate(samples: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    batch = {
        key: torch.stack([sample[key] for sample in samples])
        for key in samples[0]
    }
    # Two PP stages require at least two microbatches to exercise 1F1B rather
    # than only its warmup/cooldown path.
    return [
        {key: value.chunk(2, dim=0)[index] for key, value in batch.items()}
        for index in range(2)
    ]


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _criterion(output, microbatch):
    if isinstance(output, torch.Tensor):
        return output
    return output.loss if hasattr(output, "loss") else output["loss"]


def _validate_environment(case: AcceptanceCase, args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the full-parallel acceptance test requires CUDA")
    if dist.get_world_size() != case.world_size:
        raise RuntimeError(
            f"{case.name} requires exactly {case.world_size} ranks, got "
            f"{dist.get_world_size()}"
        )
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("--batch-size must be an even integer >= 2")
    divisor = case.cp * (2 if args.splitter == "head-tail" else 1)
    if args.sequence_length % divisor:
        raise ValueError(
            f"--sequence-length must be divisible by {divisor} for "
            f"{args.splitter} CP"
        )
    if "linear_attention" in case.layer_types:
        try:
            version = importlib_metadata.version("flash-linear-attention")
        except importlib_metadata.PackageNotFoundError as error:
            raise RuntimeError("flash-linear-attention 0.5.0 is required") from error
        if version != "0.5.0":
            raise RuntimeError(
                "flash-linear-attention 0.5.0 is required; " f"found {version}"
            )


def _run(case: AcceptanceCase, args: argparse.Namespace) -> dict[str, object]:
    rank = dist.get_rank()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    config = _config(case)
    model = from_hf_config(
        config, model_kind="language", attn_implementation="eager"
    )
    model.set_random_init()

    splitter = (
        HeadTailContextParallelSplitter()
        if args.splitter == "head-tail"
        else UniformContextParallelSplitter()
    )
    plan = ParallelizationPlan(global_ranks=list(range(case.world_size)))
    plan.parallelize(
        model,
        ParallelConfig(
            data_parallel_size=case.dp,
            pipeline_parallel_size=case.pp,
            context_parallel_size=case.cp,
            tensor_parallel_size=case.tp,
            expert_parallel_size=case.ep,
            context_parallel_splitter=splitter,
        ),
    )
    context = plan.materialize("cuda", dtype=torch.bfloat16)
    model.train()

    dataset = _TokenDataset(args.batch_size * case.dp, args.sequence_length)
    loader = context.prepare_dataloader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=_collate,
        shuffle=False,
    )
    microbatches = next(iter(loader))
    device = torch.device("cuda", torch.cuda.current_device())
    microbatches = [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in microbatch.items()
        }
        for microbatch in microbatches
    ]

    execution = CornstarchExecutionPlan()
    merged = execution.merge_modality_encoder_outputs(
        language_model=model,
        input_ids=ExecutionFuture("input_ids"),
        labels=ExecutionFuture("labels"),
        modality_token_ids={},
        encoder_outputs={},
    )
    output = execution.run_language_model(module=model, inputs=merged)
    schedule = context.create_schedule(execution, output)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    result = schedule.step(
        microbatches, _criterion, optimizer=None, return_loss=True
    )
    mesh = context.get_mesh(model)
    assert mesh is not None
    owns_loss = mesh.is_last_stage()
    local_loss_ok = (
        result["loss"] is not None
        and bool(torch.isfinite(result["loss"]).all().item())
        if owns_loss
        else result["loss"] is None
    )

    context.sync_gradients()
    gradients = [
        _local_tensor(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    local_grad_ok = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    optimizer.step()
    local_param_ok = all(
        bool(torch.isfinite(_local_tensor(parameter)).all().item())
        for parameter in model.parameters()
    )

    status = torch.tensor(
        [
            int(local_loss_ok),
            int(local_grad_ok),
            int(local_param_ok),
            int(owns_loss),
        ],
        device="cuda",
        dtype=torch.int64,
    )
    minima = status[:3].clone()
    dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    loss_owners = status[3].clone()
    dist.all_reduce(loss_owners, op=dist.ReduceOp.SUM)
    expected_loss_owners = case.dp * case.cp * case.tp * case.ep
    if not bool(minima.all().item()) or loss_owners.item() != expected_loss_owners:
        raise AssertionError(
            f"distributed step failed: minima={minima.tolist()}, "
            f"loss_owners={loss_owners.item()}, expected={expected_loss_owners}"
        )

    return {
        "case": case.name,
        "status": "passed",
        "world_size": case.world_size,
        "topology": {
            "dp": case.dp,
            "pp": case.pp,
            "cp": case.cp,
            "tp": case.tp,
            "ep": case.ep,
        },
        "splitter": args.splitter,
        "batch_size_per_dp_replica": args.batch_size,
        "sequence_length": args.sequence_length,
        "loss_owners": int(loss_owners.item()),
        "rank": rank,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate combined Qwen 5D/4D training on real GPUs."
    )
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--splitter", choices=("uniform", "head-tail"), default="head-tail")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        case = CASES[args.case]
        _validate_environment(case, args)
        report = _run(case, args)
        dist.barrier()
        if dist.get_rank() == 0:
            report.pop("rank")
            print(json.dumps(report, sort_keys=True))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
