"""Real FLA/NCCL acceptance tests for run-aware Gated DeltaNet CP."""
from __future__ import annotations

import unittest
from importlib import metadata as importlib_metadata

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_utils import FILE_SCHEMA

from tests.distributed.distributed_base import GlooDistributedTestBase

from cornstarch.distributed.context_parallel.gated_delta import (
    _flatten_local_runs,
    _restore_local_runs,
    _run_aware_convolution,
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


def _has_fla_nccl() -> bool:
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.device_count() >= 2
            and importlib_metadata.version("flash-linear-attention") == "0.5.0"
        )
    except importlib_metadata.PackageNotFoundError:
        return False


class _NCCLDistributedTestBase(GlooDistributedTestBase):
    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, pipe, **kwargs) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        torch.cuda.set_device(rank)
        dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{file_name}",
            backend="nccl",
            world_size=self.world_size,
            rank=rank,
        )
        try:
            self.reset_seed()
            self.run_test(test_name, pipe)
        finally:
            dist.destroy_process_group()


@unittest.skipUnless(_has_fla_nccl(), "requires two CUDA GPUs and FLA 0.5.0")
class TestGatedDeltaContextParallelNCCL(_NCCLDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _gather_global(
        self, local: torch.Tensor, offsets: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        output = local.new_zeros(local.shape[0], 8, *local.shape[2:])
        for rank, rank_offsets in enumerate(offsets):
            output[:, rank_offsets] = gathered[rank]
        return output

    def _operator_parity(self, splitter, document_ids: torch.Tensor) -> None:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        device = torch.device("cuda", self.rank)
        mask = document_ids >= 0
        offsets = tuple(splitter.offsets_for_size(mask.cpu(), self.world_size))
        metadata = build_gated_delta_metadata(
            offsets, mask.cpu(), document_ids=document_ids.cpu()
        )
        heads, key_dim, value_dim = 2, 16, 12
        generator = torch.Generator(device=device).manual_seed(123)
        shapes = (
            (1, 8, heads, key_dim),
            (1, 8, heads, key_dim),
            (1, 8, heads, value_dim),
            (1, 8, heads),
            (1, 8, heads),
        )
        full = [
            torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
            for shape in shapes
        ]
        full[3] = -full[3].float().softplus().to(torch.bfloat16)
        full[4] = full[4].sigmoid()
        valid = int(mask.sum())
        full_cu = [0]
        for index in range(1, valid):
            if document_ids[0, index] != document_ids[0, index - 1]:
                full_cu.append(index)
        full_cu.append(valid)

        reference = [item.detach().requires_grad_(True) for item in full]
        reference_output, _ = chunk_gated_delta_rule(
            *(item[:, :valid] for item in reference),
            cu_seqlens=torch.tensor(full_cu, dtype=torch.int32, device=device),
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        reference_output.float().sum().backward()

        rank_offsets = offsets[self.rank].to(device)
        local = [
            item[:, rank_offsets].detach().requires_grad_(True) for item in full
        ]
        local_runs = metadata.local_runs(self.rank)
        lengths = [run.length for run in local_runs] or [1]
        cu_cpu = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        )
        context = CornstarchRunAwareFLACPContext(
            group=dist.group.WORLD,
            cu_seqlens=cu_cpu.to(device),
            cu_seqlens_cpu=cu_cpu,
            metadata=metadata,
            local_run_indices=(
                tuple(run.index for run in local_runs) if local_runs else (-1,)
            ),
        )
        install_run_aware_fla_dispatch()
        packed = [_flatten_local_runs(item, metadata, self.rank) for item in local]
        output, _ = chunk_gated_delta_rule(
            *packed,
            cp_context=context,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        output.float().sum().backward()
        restored = _restore_local_runs(output, local[2], metadata, self.rank)
        global_output = self._gather_global(restored, offsets)
        torch.testing.assert_close(
            global_output[:, :valid], reference_output, atol=2e-2, rtol=2e-2
        )
        for local_tensor, reference_tensor in zip(local, reference):
            global_grad = self._gather_global(local_tensor.grad, offsets)
            torch.testing.assert_close(
                global_grad[:, :valid],
                reference_tensor.grad[:, :valid],
                atol=3e-2,
                rtol=3e-2,
            )

    def test_uniform_and_headtail_packed_padding_forward_backward(self) -> None:
        packed = torch.tensor([[0, 0, 0, 1, 1, 1, 1, -1]], device="cuda")
        for splitter in (
            UniformContextParallelSplitter(),
            HeadTailContextParallelSplitter(),
        ):
            self._operator_parity(splitter, packed)

    def test_headtail_empty_lane_and_bidirectional_convolution_halos(self) -> None:
        document_ids = torch.tensor([[0, 0, -1, -1, -1, -1, -1, -1]], device="cuda")
        self._operator_parity(HeadTailContextParallelSplitter(), document_ids)

        device = torch.device("cuda", self.rank)
        mask = torch.ones(1, 8, dtype=torch.bool)
        splitter = HeadTailContextParallelSplitter()
        offsets = tuple(splitter.offsets_for_size(mask, self.world_size))
        metadata = build_gated_delta_metadata(offsets, mask)
        full = torch.randn(1, 3, 8, device=device, requires_grad=True)
        weight = torch.randn(3, 3, device=device, requires_grad=True)
        reference = F.silu(
            F.conv1d(F.pad(full, (2, 0)), weight[:, None], groups=3)
        )
        rank_offsets = offsets[self.rank].to(device)
        local = full.detach()[:, :, rank_offsets].requires_grad_(True)
        local_weight = weight.detach().clone().requires_grad_(True)
        output = _run_aware_convolution(
            local, local_weight, None, metadata, dist.group.WORLD
        )
        gathered = [torch.empty_like(output) for _ in range(self.world_size)]
        dist.all_gather(gathered, output)
        restored = torch.zeros_like(reference)
        for rank, rank_offsets in enumerate(offsets):
            restored[:, :, rank_offsets] = gathered[rank]
        torch.testing.assert_close(restored, reference, atol=2e-5, rtol=2e-5)

        dout = torch.randn_like(reference)
        reference.backward(dout)
        output.backward(dout[:, :, offsets[self.rank].to(device)])
        gathered_grads = [torch.empty_like(local.grad) for _ in range(self.world_size)]
        dist.all_gather(gathered_grads, local.grad)
        restored_grad = torch.zeros_like(full)
        for rank, rank_offsets in enumerate(offsets):
            restored_grad[:, :, rank_offsets] = gathered_grads[rank]
        dist.all_reduce(local_weight.grad, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(
            restored_grad, full.grad, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            local_weight.grad, weight.grad, atol=2e-5, rtol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
