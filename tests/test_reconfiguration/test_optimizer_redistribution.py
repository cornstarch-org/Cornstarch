"""Tests for ReconfigurationExecutor.redistribute_optimizer_states()."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import instantiate_parametrized_tests

from cornstarch.reconfiguration.data_structures import LayerOwnership
from cornstarch.reconfiguration.executor import ReconfigurationExecutor

from ..distributed_base import GlooDistributedTestBase


class SimpleModel(nn.Module):
    """Four named parameters used for ownership tests."""

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.randn(10, 10))
        self.param2 = nn.Parameter(torch.randn(20, 20))
        self.param3 = nn.Parameter(torch.randn(30, 30))
        self.param4 = nn.Parameter(torch.randn(40, 40))


class _OptimizerStub:
    """Minimal optimizer stub that exposes the ColossalAI OptimizerWrapper API
    required by ``ReconfigurationExecutor.redistribute_optimizer_states()``.

    Wraps a plain ``torch.optim.Adam`` and exposes:
    - ``optim`` — the underlying optimizer (has ``.state`` and ``.param_groups``)
    - ``param_info`` — maps ``id(param) → param_group_index``
    - ``get_master_to_working_map()`` → None  (no AMP master params)
    """

    def __init__(self, model: nn.Module):
        params = list(model.parameters())
        self.optim = torch.optim.Adam(params, lr=1e-3)
        self.param_info = {
            "param2id": {id(p): i for i, p in enumerate(params)},
        }

    def get_master_to_working_map(self):
        return None


@instantiate_parametrized_tests
class OptimizerRedistributionTests(GlooDistributedTestBase):
    """Tests for optimizer-state redistribution via ReconfigurationExecutor."""

    @property
    def world_size(self) -> int:
        return 4

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_source_ownership(self, rank):
        """Source: ranks 0-1 → param1+param2; ranks 2-3 → param3+param4."""
        all_params = ["param1", "param2", "param3", "param4"]
        if rank in (0, 1):
            owned = ["param1", "param2"]
        else:
            owned = ["param3", "param4"]
        placeholder = {p: p not in owned for p in all_params}
        return LayerOwnership(rank=rank, layer_names=owned, is_placeholder=placeholder)

    def _build_target_ownership(self, rank):
        """Target: ranks 0-1 → param1+param3; ranks 2-3 → param2+param4."""
        all_params = ["param1", "param2", "param3", "param4"]
        if rank in (0, 1):
            owned = ["param1", "param3"]
        else:
            owned = ["param2", "param4"]
        placeholder = {p: p not in owned for p in all_params}
        return LayerOwnership(rank=rank, layer_names=owned, is_placeholder=placeholder)

    def _gather_all_ownership(self, local):
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, {
            "rank": local.rank, "layer_names": local.layer_names,
            "is_placeholder": local.is_placeholder,
        })
        return {
            d["rank"]: LayerOwnership(**d)
            for d in all_data if d is not None
        }

    # ------------------------------------------------------------------
    # test_optimizer_states_follow_params
    # ------------------------------------------------------------------

    def test_optimizer_states_follow_params(self):
        """Optimizer states (exp_avg, exp_avg_sq) move with their parameters.

        Source layout: ranks 0-1 hold param1/param2; ranks 2-3 hold param3/param4.
        Target layout: ranks 0-1 hold param1/param3; ranks 2-3 hold param2/param4.
        """
        torch.manual_seed(self.rank)
        model = SimpleModel()

        # Seed each param with a known value so we can verify after redistribution.
        for name, param in model.named_parameters():
            param.data.fill_(float(hash(name) % 100) / 10.0)

        optimizer_stub = _OptimizerStub(model)

        # Run one optimizer step to populate exp_avg / exp_avg_sq.
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        optimizer_stub.optim.step()
        optimizer_stub.optim.zero_grad()

        # Record original states BEFORE redistribution.
        original_states = {}
        for param, state in optimizer_stub.optim.state.items():
            name = None
            for n, p in model.named_parameters():
                if p is param:
                    name = n
                    break
            if name is not None:
                original_states[name] = {k: v.clone() for k, v in state.items() if isinstance(v, torch.Tensor)}

        # Broadcast originals so every rank has a reference.
        for name in ["param1", "param2", "param3", "param4"]:
            if name in original_states:
                src = 0 if name in ("param1", "param2") else 2
            else:
                src = 0 if name in ("param1", "param2") else 2
            # Get the state tensors that rank `src` has.
            for key in ["exp_avg", "exp_avg_sq"]:
                # All ranks allocate a buffer to receive the broadcast.
                param = getattr(model, name)
                buf = torch.zeros_like(param)
                if name in original_states and key in original_states[name]:
                    buf.copy_(original_states[name][key])
                dist.broadcast(buf, src=src)
                if name not in original_states:
                    original_states[name] = {}
                original_states[name][key] = buf.clone()

        # Build ownership maps.
        src_local = self._build_source_ownership(self.rank)
        tgt_local = self._build_target_ownership(self.rank)
        source_ownership = self._gather_all_ownership(src_local)
        target_ownership = self._gather_all_ownership(tgt_local)

        # Capture parameter snapshot BEFORE execute() replaces param objects.
        param_snapshot = {n: p for n, p in model.named_parameters()}

        # Redistribute model parameters.
        executor = ReconfigurationExecutor(model)
        executor.execute(source_ownership, target_ownership)

        # Redistribute optimizer states using the pre-execution snapshot so
        # identity-based state lookup still works.
        executor.redistribute_optimizer_states(
            optimizer_stub, source_ownership, target_ownership,
            param_snapshot=param_snapshot,
        )

        # Verify: each rank should now have optimizer states for its target params.
        # After execute() the param objects changed, so look up state by the
        # pre-snapshot param (which is what the state dict is still keyed by).
        target_params = target_ownership[self.rank].layer_names
        for name in target_params:
            original_param = param_snapshot[name]
            found_state = optimizer_stub.optim.state.get(original_param)
            assert found_state is not None, (
                f"Rank {self.rank}: no optimizer state found for {name}"
            )
            for key in ["exp_avg", "exp_avg_sq"]:
                assert key in found_state, (
                    f"Rank {self.rank}: missing '{key}' in optimizer state for {name}"
                )
                assert torch.allclose(found_state[key], original_states[name][key], atol=1e-5), (
                    f"Rank {self.rank}: optimizer state '{key}' for {name} does not match original"
                )

        print(f"Rank {self.rank}: test_optimizer_states_follow_params passed")

    # ------------------------------------------------------------------
    # test_identity_optimizer_redistribution
    # ------------------------------------------------------------------

    def test_identity_optimizer_redistribution(self):
        """When source == target, optimizer states are unchanged."""
        torch.manual_seed(0)
        model = SimpleModel()
        optimizer_stub = _OptimizerStub(model)

        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        optimizer_stub.optim.step()
        optimizer_stub.optim.zero_grad()

        # Save original states keyed by param object BEFORE any redistribution.
        param_snapshot = {n: p for n, p in model.named_parameters()}
        original = {
            n: {k: v.clone() for k, v in optimizer_stub.optim.state[p].items()
                if isinstance(v, torch.Tensor)}
            for n, p in param_snapshot.items()
            if p in optimizer_stub.optim.state
        }

        # Build source ownership (same as target).
        if self.rank in (0, 1):
            owned = ["param1", "param2"]
        else:
            owned = ["param3", "param4"]
        all_params = ["param1", "param2", "param3", "param4"]
        local = LayerOwnership(
            rank=self.rank,
            layer_names=owned,
            is_placeholder={p: p not in owned for p in all_params},
        )
        all_data = [None] * self.world_size
        dist.all_gather_object(all_data, {"rank": local.rank, "layer_names": local.layer_names, "is_placeholder": local.is_placeholder})
        ownership = {d["rank"]: LayerOwnership(**d) for d in all_data if d is not None}

        executor = ReconfigurationExecutor(model)
        executor.redistribute_optimizer_states(
            optimizer_stub, ownership, ownership, param_snapshot=param_snapshot
        )

        # States must be unchanged.
        for n, p in param_snapshot.items():
            if n not in original:
                continue
            state = optimizer_stub.optim.state.get(p, {})
            for key, orig_val in original[n].items():
                assert key in state
                assert torch.allclose(state[key], orig_val, atol=1e-6), (
                    f"Rank {self.rank}: state '{key}' for {n} changed unexpectedly"
                )

        print(f"Rank {self.rank}: test_identity_optimizer_redistribution passed")
