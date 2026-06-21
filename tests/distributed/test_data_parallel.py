"""Tests for GradientSynchronizer and bucketed all-reduce (gloo backend)."""
import unittest

import torch
import torch.nn as nn
import torch.distributed as dist

from tests.distributed.distributed_base import GlooDistributedTestBase
from cornstarch.distributed.process_group_mesh import ModalProcessGroupMesh
from cornstarch.distributed.data_parallel import GradientSynchronizer


class SimpleMLP(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc(x)


class TestGradientSynchronizer(GlooDistributedTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def _make_mesh(self) -> ModalProcessGroupMesh:
        return ModalProcessGroupMesh(
            device_type="cpu",
            global_ranks=[0, 1],
            dp_size=2,
            cp_size=1,
            tp_size=1,
            num_pp_stages=1,
        )

    def test_gradient_sync(self):
        """Both ranks should produce identical averaged gradients after sync."""
        mesh = self._make_mesh()
        torch.manual_seed(42)
        model = SimpleMLP(dim=8)

        grad_sync = GradientSynchronizer(mesh.dp_group)
        grad_sync.register(model)

        rank = dist.get_rank()
        torch.manual_seed(rank)
        x = torch.randn(4, 8)

        model(x).sum().backward()
        grad_sync.sync()

        local_grad = model.fc.weight.grad.clone()
        gathered = [torch.zeros_like(local_grad) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_grad)

        self.assertTrue(
            torch.allclose(gathered[0], gathered[1]),
            f"Gradients differ: {gathered[0]} vs {gathered[1]}",
        )

    def test_weight_update_consistency(self):
        """Parameters should remain identical across ranks after an optimizer step."""
        mesh = self._make_mesh()
        torch.manual_seed(42)
        model = SimpleMLP(dim=8)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        grad_sync = GradientSynchronizer(mesh.dp_group)
        grad_sync.register(model)

        rank = dist.get_rank()
        torch.manual_seed(rank)
        x = torch.randn(4, 8)

        optimizer.zero_grad()
        model(x).sum().backward()
        grad_sync.sync()
        optimizer.step()

        local_w = model.fc.weight.data.clone()
        gathered = [torch.zeros_like(local_w) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_w)
        self.assertTrue(torch.allclose(gathered[0], gathered[1]))

    def test_multi_model_sync(self):
        """Sync with multiple registered models averages all their gradients."""
        mesh = self._make_mesh()
        torch.manual_seed(42)
        model_a = SimpleMLP(dim=8)
        model_b = SimpleMLP(dim=4)

        grad_sync = GradientSynchronizer(mesh.dp_group)
        grad_sync.register(model_a)
        grad_sync.register(model_b)

        rank = dist.get_rank()
        torch.manual_seed(rank)
        xa = torch.randn(4, 8)
        xb = torch.randn(4, 4)

        (model_a(xa).sum() + model_b(xb).sum()).backward()
        grad_sync.sync()

        for model in [model_a, model_b]:
            local_grad = model.fc.weight.grad.clone()
            gathered = [torch.zeros_like(local_grad) for _ in range(self.world_size)]
            dist.all_gather(gathered, local_grad)
            self.assertTrue(
                torch.allclose(gathered[0], gathered[1]),
                f"Gradients differ for {model}",
            )


if __name__ == "__main__":
    unittest.main()
