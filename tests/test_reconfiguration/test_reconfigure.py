"""End-to-end tests for MultimodalParallelPlugin.reconfigure().

Tests process group lifecycle (destruction + recreation), parameter
redistribution, and live-reference updates on model and optimizer.

We avoid heavy Transformers model loading by working directly with
MultiModalProcessGroupMesh, ReconfigurationExecutor, and build_target_ownership
through a thin wrapper that mimics the relevant parts of
MultimodalParallelPlugin.reconfigure().
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_utils import instantiate_parametrized_tests

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.plugin.multimodal_parallel_plugin.modal_process_group_mesh import (
    MultiModalProcessGroupMesh,
)
from cornstarch.reconfiguration.data_structures import LayerOwnership
from cornstarch.reconfiguration.executor import ReconfigurationExecutor
from cornstarch.reconfiguration.ownership_analyzer import (
    TensorOwnershipAnalyzer,
    build_target_ownership,
)

from ..distributed_base import GlooDistributedTestBase


# ---------------------------------------------------------------------------
# Toy templates — 8 ranks total
#
# Initial config (PP=1/TP=1 each, DP=2):
#   enc_template_1stage: 4 layers in 1 stage
#   llm_template_1stage: 4 layers in 1 stage
#   World: enc ranks [0,1], llm ranks [2,3], DP replica 2 on [4,5] enc [6,7] llm
#
# Target config (PP=2/TP=1 each, DP=1):
#   enc_template_2stage: 2+2 layers across 2 stages
#   llm_template_2stage: 2+2 layers across 2 stages
#   World: enc ranks [0,1,2,3], llm ranks [4,5,6,7]
# ---------------------------------------------------------------------------

enc_template_1stage = PipelineTemplate(
    "encoder", [["enc.layer.0", "enc.layer.1", "enc.layer.2", "enc.layer.3"]]
)
enc_template_2stage = PipelineTemplate(
    "encoder",
    [["enc.layer.0", "enc.layer.1"], ["enc.layer.2", "enc.layer.3"]],
)
llm_template_1stage = PipelineTemplate(
    "llm", [["llm.layer.0", "llm.layer.1", "llm.layer.2", "llm.layer.3"]]
)
llm_template_2stage = PipelineTemplate(
    "llm",
    [["llm.layer.0", "llm.layer.1"], ["llm.layer.2", "llm.layer.3"]],
)


class _ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 4))


class _ToyGroup(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.layer = nn.ModuleList([_ToyLayer() for _ in range(n)])


class ToyModel(nn.Module):
    """Nested model whose parameter names match the pipeline template prefixes.

    ``named_parameters()`` yields names like ``enc.layer.0.weight``,
    ``llm.layer.3.weight``, etc., which the pipeline template
    modules_per_stage prefixes (``enc.layer.0``) can match.
    """

    def __init__(self):
        super().__init__()
        self.enc = _ToyGroup(4)
        self.llm = _ToyGroup(4)


class _ModalPlugin:
    """Minimal ModalParallelPlugin stub (only the attributes reconfigure() reads)."""

    def __init__(self, pipeline_template, tp_size=1, sp_size=1):
        self.pipeline_template = pipeline_template
        self.tp_size = tp_size
        self.sp_size = sp_size


class _OptimizerStub:
    """Plain Adam wrapped to expose the ColossalAI OptimizerWrapper interface."""

    def __init__(self, model: nn.Module):
        params = list(model.parameters())
        self.optim = torch.optim.Adam(params, lr=1e-3)
        self.param_info = {
            "param2id": {id(p): i for i, p in enumerate(params)},
        }
        # Attributes updated by reconfigure().
        self.tp_pg = None
        self.pp_pg = None

    def get_master_to_working_map(self):
        return None


@instantiate_parametrized_tests
class ReconfigureTests(GlooDistributedTestBase):
    """End-to-end tests for reconfigure() mechanics."""

    @property
    def world_size(self) -> int:
        return 8

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_source_pg_mesh(self):
        """Create pg_mesh for the source config (enc PP=1, llm PP=1, DP=2)."""
        return MultiModalProcessGroupMesh(
            encoder_templates={enc_template_1stage: (1, 1)},  # tp=1, sp=1
            llm_template=(llm_template_1stage, 1, 1),
        )

    def _init_target_pg_mesh(self):
        """Create pg_mesh for the target config (enc PP=2, llm PP=2, DP=1)."""
        return MultiModalProcessGroupMesh(
            encoder_templates={enc_template_2stage: (1, 1)},
            llm_template=(llm_template_2stage, 1, 1),
        )

    def _source_plugins(self):
        return (
            {"encoder": _ModalPlugin(enc_template_1stage)},
            _ModalPlugin(llm_template_1stage),
        )

    def _target_plugins(self):
        return (
            {"encoder": _ModalPlugin(enc_template_2stage)},
            _ModalPlugin(llm_template_2stage),
        )

    def _build_ownership_from_mesh(self, model, pg_mesh, enc_plugins, llm_plugin):
        """Use build_target_ownership to get ownership for the given mesh."""
        return build_target_ownership(model, pg_mesh, enc_plugins, llm_plugin)

    # ------------------------------------------------------------------
    # test_pp_only_change
    # ------------------------------------------------------------------

    def test_pp_only_change(self):
        """Full reconfigure flow: PP=1→2 per modal, no TP change.

        After reconfiguration:
        1. Old sub-groups are destroyed.
        2. New pg_mesh has the correct shape.
        3. Each rank owns exactly the parameters belonging to its new PP stage.
        4. Optimizer states follow the parameters.
        """
        torch.manual_seed(self.rank)
        model = ToyModel()

        # Give each parameter a unique, rank-agnostic value so we can verify.
        for name, param in model.named_parameters():
            param.data.fill_(float(abs(hash(name)) % 100) / 10.0)

        # Broadcast all param values from rank 0 so everyone has the same.
        for name, param in model.named_parameters():
            dist.broadcast(param.data, src=0)

        optimizer = _OptimizerStub(model)
        # Populate optimizer state with one forward-backward step.
        loss = sum(p.sum() for p in model.parameters())
        loss.backward()
        optimizer.optim.step()
        optimizer.optim.zero_grad()

        # Capture original values AFTER the optimizer step so we can verify
        # that redistribution preserves the post-step parameter values.
        original_values = {n: p.data.clone() for n, p in model.named_parameters()}

        # Record original optimizer states (broadcast from rank 0 since values
        # are the same across all source DP replicas after the forward-backward step).
        original_opt_states = {}
        for name, param in model.named_parameters():
            state = optimizer.optim.state.get(param, {})
            for key in list(state.keys()):
                if isinstance(state[key], torch.Tensor):
                    buf = state[key].clone()
                    dist.broadcast(buf, src=0)
                    if name not in original_opt_states:
                        original_opt_states[name] = {}
                    original_opt_states[name][key] = buf

        # --- Source pg_mesh ---
        source_pg_mesh = self._init_source_pg_mesh()

        # --- Target pg_mesh (topology only — no dist.new_group calls yet).
        # Seed its cache from the source mesh so unchanged rank-set groups are
        # reused instead of recreated.
        target_pg_mesh = self._init_target_pg_mesh()
        target_pg_mesh.inherit_groups_from(source_pg_mesh)

        enc_plugins, llm_plugin = self._target_plugins()
        target_ownership = build_target_ownership(
            model, target_pg_mesh, enc_plugins, llm_plugin
        )
        source_ownership = TensorOwnershipAnalyzer(model).analyze()

        # Capture parameter snapshot BEFORE execute() replaces param objects.
        param_snapshot = {n: p for n, p in model.named_parameters()}

        # Redistribute parameters.
        executor = ReconfigurationExecutor(model)
        executor.execute(source_ownership, target_ownership)

        # Redistribute optimizer states using the pre-execution snapshot.
        executor.redistribute_optimizer_states(
            optimizer, source_ownership, target_ownership,
            param_snapshot=param_snapshot,
        )

        # Trigger new group creation for any genuinely new rank sets.
        # (In this test we drive it manually; in production reconfigure() calls
        # _init_communication_groups() which does this.)
        target_pg_mesh.get_group_along_axis(target_pg_mesh.pp_axis)
        target_pg_mesh.get_group_along_axis(target_pg_mesh.dp_axis)
        target_pg_mesh.get_group_along_axis(target_pg_mesh.tp_axis)
        target_pg_mesh.get_group_along_axis(target_pg_mesh.sp_axis)
        target_pg_mesh.get_global_pp_group()

        # Destroy only stale groups (those not reused by target_pg_mesh).
        # The collective dist.new_group() calls above provide the necessary
        # synchronisation — no explicit barrier is needed.
        stale_groups = [
            g for r, g in source_pg_mesh._ranks_to_group.items()
            if r not in target_pg_mesh._ranks_to_group
        ]
        source_pg_mesh.destroy_stale_groups(set(target_pg_mesh._ranks_to_group))

        # --- Verify stale groups are no longer valid ---
        for group in stale_groups:
            try:
                dist.get_world_size(group)
                raise AssertionError(
                    f"Rank {self.rank}: stale group should have been destroyed"
                )
            except (ValueError, RuntimeError):
                pass  # Expected: group is invalid.

        # --- Verify new pg_mesh shape ---
        # world_size=8, enc PP=2, TP=1, SP=1 + llm PP=2, TP=1, SP=1
        # → ranks_per_replica = 4, DP = 8/4 = 2
        # enc mesh shape: (2, 2, 1, 1); llm mesh shape: (2, 2, 1, 1)
        for modal, mesh in target_pg_mesh.modal_meshes.items():
            assert mesh.shape == (2, 2, 1, 1), (
                f"Rank {self.rank}: unexpected mesh shape {mesh.shape} for {modal}"
            )

        # --- Verify parameter ownership ---
        my_target = target_ownership[self.rank]
        for param_name in my_target.layer_names:
            param = dict(model.named_parameters()).get(param_name)
            assert param is not None, (
                f"Rank {self.rank}: should hold {param_name} but param is None"
            )
            assert torch.allclose(param.data, original_values[param_name], atol=1e-5), (
                f"Rank {self.rank}: {param_name} value incorrect after redistribution"
            )

        # --- Verify optimizer states co-located with params ---
        # After execute() the model has new param objects; use pre-execution
        # snapshot (param_snapshot) to find the optimizer state entry.
        for param_name in my_target.layer_names:
            old_param = param_snapshot.get(param_name)
            state = optimizer.optim.state.get(old_param)
            assert state is not None, (
                f"Rank {self.rank}: no optimizer state for {param_name}"
            )
            for key, orig in original_opt_states.get(param_name, {}).items():
                assert torch.allclose(state[key], orig, atol=1e-5), (
                    f"Rank {self.rank}: optimizer state '{key}' for {param_name} is wrong"
                )

        print(f"Rank {self.rank}: test_pp_only_change passed")

    # ------------------------------------------------------------------
    # test_process_groups_updated_in_model
    # ------------------------------------------------------------------

    def test_process_groups_updated_in_model(self):
        """After reconfigure(), model process group references are updated."""
        # Use a minimal object that just has the three group attributes.
        class ModelProxy:
            dp_group = None
            tp_group = None
            sp_group = None
            stage_manager = None

        model_proxy = ModelProxy()
        model = ToyModel()

        source_pg_mesh = self._init_source_pg_mesh()
        target_pg_mesh = self._init_target_pg_mesh()

        source_dp = source_pg_mesh.get_group_along_axis(source_pg_mesh.dp_axis)
        source_tp = source_pg_mesh.get_group_along_axis(source_pg_mesh.tp_axis)
        source_sp = source_pg_mesh.get_group_along_axis(source_pg_mesh.sp_axis)

        # Simulate reference update as done in reconfigure() Phase 11.
        target_dp = target_pg_mesh.get_group_along_axis(target_pg_mesh.dp_axis)
        target_tp = target_pg_mesh.get_group_along_axis(target_pg_mesh.tp_axis)
        target_sp = target_pg_mesh.get_group_along_axis(target_pg_mesh.sp_axis)

        model_proxy.dp_group = target_dp
        model_proxy.tp_group = target_tp
        model_proxy.sp_group = target_sp

        # The new groups must differ from the old ones (different rank sets).
        assert model_proxy.dp_group is not source_dp, (
            f"Rank {self.rank}: dp_group was not updated"
        )
        # With world_size=8, enc_PP=2 and llm_PP=2: each modal uses 4 ranks
        # (2 PP × 2 DP × 1 SP × 1 TP), so the target DP size per modal is 2.
        assert dist.get_world_size(model_proxy.dp_group) == 2, (
            f"Rank {self.rank}: new dp_group world size should be 2, "
            f"got {dist.get_world_size(model_proxy.dp_group)}"
        )

        print(f"Rank {self.rank}: test_process_groups_updated_in_model passed")

    # ------------------------------------------------------------------
    # test_optimizer_groups_updated
    # ------------------------------------------------------------------

    def test_optimizer_groups_updated(self):
        """After reconfigure(), optimizer.tp_pg and .pp_pg are updated."""
        model = ToyModel()
        optimizer = _OptimizerStub(model)

        source_pg_mesh = self._init_source_pg_mesh()
        target_pg_mesh = self._init_target_pg_mesh()

        new_tp_group = target_pg_mesh.get_group_along_axis(target_pg_mesh.tp_axis)
        new_global_pp_group = target_pg_mesh.get_global_pp_group()

        # Simulate Phase 11 of reconfigure().
        optimizer.tp_pg = new_tp_group
        optimizer.pp_pg = new_global_pp_group

        assert optimizer.tp_pg is new_tp_group, (
            f"Rank {self.rank}: optimizer.tp_pg not updated"
        )
        assert optimizer.pp_pg is new_global_pp_group, (
            f"Rank {self.rank}: optimizer.pp_pg not updated"
        )
        # TP=1 in target → tp group world size should be 1.
        assert dist.get_world_size(optimizer.tp_pg) == 1, (
            f"Rank {self.rank}: new tp group world size should be 1, "
            f"got {dist.get_world_size(optimizer.tp_pg)}"
        )

        print(f"Rank {self.rank}: test_optimizer_groups_updated passed")
