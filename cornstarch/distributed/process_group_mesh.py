from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh


class ModalProcessGroupMesh:
    """Per-modality 5-D process group mesh covering DP, PP, CP, TP, and EP.

    Each modality in a multimodal model owns its own ``ModalProcessGroupMesh``
    that spans only the global ranks assigned to that modality.  All five
    parallelism dimensions live in a single ``DeviceMesh`` so that every rank
    in the modality participates in the same construction call — this avoids
    the deadlocks that occur when different rank subsets try to create
    independent meshes under gloo.

    Expert parallelism (EP) is a first-class mesh axis here, not a separate
    whole-world group: it composes with DP/CP/TP/PP exactly like the other
    dimensions.  Expert weights are sharded along the ``"ep"`` axis while
    attention / non-expert weights are replicated across it; conversely the
    expert shards are replicated across the ``"dp"`` axis.  Promoting EP into
    the mesh is what lets a MoE model be expert-parallel *and* tensor- or
    pipeline-parallel at the same time.

    Sub-meshes and process groups for individual dimensions are extracted via
    ``DeviceMesh.__getitem__`` / ``.get_group()``, which are local operations
    that require no collectives.  Each dimension exposes both forms
    (``dp_mesh``/``dp_group``, ``cp_mesh``/``cp_group``, ``tp_mesh``/``tp_group``,
    ``pp_mesh``/``pp_group``, ``ep_mesh``/``ep_group``); the built-in strategies
    consume whichever form their API needs (``tp_mesh`` for TP, ``cp_group`` for
    CP, ``dp_group`` for DP sync, ``ep_group`` for EP all-to-all).  The PP stage
    index for the current rank is derived from its position along the ``"pp"``
    dimension rather than being passed in explicitly.

    Rank layout
    -----------
    ``global_ranks`` is interpreted in ``(dp, pp, cp, tp, ep)`` order, replica
    major and EP minor — i.e. reshaped to
    ``(dp_size, num_pp_stages, cp_size, tp_size, ep_size)``.  EP being the
    innermost (fastest-varying) axis means the ranks that hold the different
    shards of one MoE layer are contiguous, which keeps the all-to-all expert
    exchange local.  The PP routing math groups by the per-stage inner block
    ``cp_size * tp_size * ep_size`` so a rank always communicates with the rank
    holding the same ``(cp, tp, ep)`` position in the adjacent stage.
    """

    def __init__(
        self,
        *,
        device_type: str,
        global_ranks: list[int],
        dp_size: int,
        cp_size: int,
        tp_size: int,
        num_pp_stages: int,
        ep_size: int = 1,
    ) -> None:
        """Build a 5-D mesh for one modality from its rank assignments.

        ``global_ranks`` must contain exactly ``dp_size × num_pp_stages ×
        cp_size × tp_size × ep_size`` entries in replica-major / EP-minor order.
        All ranks that belong to this modality must call this constructor with
        the same arguments.
        """
        expected = dp_size * num_pp_stages * cp_size * tp_size * ep_size
        if len(global_ranks) != expected:
            raise ValueError(
                f"global_ranks length {len(global_ranks)} does not match "
                f"dp({dp_size}) × pp({num_pp_stages}) × cp({cp_size}) × "
                f"tp({tp_size}) × ep({ep_size}) = {expected}."
            )

        self._dp_size = dp_size
        self._cp_size = cp_size
        self._tp_size = tp_size
        self._ep_size = ep_size
        self._num_stages = num_pp_stages
        self._global_ranks = global_ranks
        self._my_rank = dist.get_rank()

        mesh_tensor = torch.tensor(global_ranks, dtype=torch.int).reshape(
            dp_size, num_pp_stages, cp_size, tp_size, ep_size
        )
        self._device_mesh: DeviceMesh = DeviceMesh(
            device_type,
            mesh_tensor,
            mesh_dim_names=("dp", "pp", "cp", "tp", "ep"),
        )

        # Derive this rank's PP stage from its position in the flat layout.
        flat_pos = global_ranks.index(self._my_rank)
        inner = cp_size * tp_size * ep_size
        self._stage = (flat_pos % (num_pp_stages * inner)) // inner

    # ------------------------------------------------------------------
    # Per-dimension accessors
    #
    # Every mesh dimension exposes both its DeviceMesh sub-mesh and its
    # ProcessGroup so any dimension's collectives are reachable, even though
    # each built-in strategy consumes only the form its API requires:
    #   - tp_mesh  -> apply_tensor_parallel (parallelize_module wants a mesh)
    #   - cp_group -> CP all-gather attention and sequence splitters
    #   - dp_group -> GradientSynchronizer all-reduce
    #   - ep_group -> apply_expert_parallel all-to-all dispatch
    # Pipeline parallelism routes by explicit ranks (get_prev_ranks /
    # get_next_ranks) rather than a collective, so pp_mesh / pp_group are
    # provided only for pipeline-wide collectives a caller might add.
    # ------------------------------------------------------------------

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._device_mesh

    @property
    def dp_mesh(self) -> DeviceMesh:
        return self._device_mesh["dp"]

    @property
    def cp_mesh(self) -> DeviceMesh:
        return self._device_mesh["cp"]

    @property
    def tp_mesh(self) -> DeviceMesh:
        return self._device_mesh["tp"]

    @property
    def pp_mesh(self) -> DeviceMesh:
        return self._device_mesh["pp"]

    @property
    def ep_mesh(self) -> DeviceMesh:
        return self._device_mesh["ep"]

    @property
    def dp_group(self) -> dist.ProcessGroup:
        return self._device_mesh["dp"].get_group()

    @property
    def cp_group(self) -> dist.ProcessGroup:
        return self._device_mesh["cp"].get_group()

    @property
    def tp_group(self) -> dist.ProcessGroup:
        return self._device_mesh["tp"].get_group()

    @property
    def pp_group(self) -> dist.ProcessGroup:
        return self._device_mesh["pp"].get_group()

    @property
    def ep_group(self) -> dist.ProcessGroup:
        return self._device_mesh["ep"].get_group()

    # ------------------------------------------------------------------
    # Sizes
    # ------------------------------------------------------------------

    @property
    def dp_size(self) -> int:
        return self._dp_size

    @property
    def cp_size(self) -> int:
        return self._cp_size

    @property
    def tp_size(self) -> int:
        return self._tp_size

    @property
    def ep_size(self) -> int:
        return self._ep_size

    # ------------------------------------------------------------------
    # Pipeline stage semantics
    # ------------------------------------------------------------------

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def num_stages(self) -> int:
        return self._num_stages

    def is_first_stage(self) -> bool:
        return self._stage == 0

    def is_last_stage(self) -> bool:
        return self._stage == self._num_stages - 1

    def get_prev_ranks(self) -> list[int]:
        """Return the global ranks that send forward activations to this rank.

        Empty on the first stage (no predecessor within the modality).
        """
        if self.is_first_stage():
            return []
        prev_stage_ranks = self._ranks_at_stage(self._stage - 1)
        my_flat = self._my_flat_index_in_stage(self._stage)
        return [prev_stage_ranks[my_flat]]

    def get_next_ranks(self) -> list[int]:
        """Return the global ranks that receive forward activations from this rank.

        Empty on the last stage (no successor within the modality).
        """
        if self.is_last_stage():
            return []
        next_stage_ranks = self._ranks_at_stage(self._stage + 1)
        my_flat = self._my_flat_index_in_stage(self._stage)
        return [next_stage_ranks[my_flat]]

    def distribute_layers(self, total_layers: int) -> tuple[int, int]:
        """Return the ``(start, end)`` layer slice for this PP stage."""
        base = total_layers // self._num_stages
        remainder = total_layers % self._num_stages
        start = self._stage * base + min(self._stage, remainder)
        end = start + base + (1 if self._stage < remainder else 0)
        return start, end

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inner(self) -> int:
        """Per-stage inner block size: the ranks sharing one PP stage's work."""
        return self._cp_size * self._tp_size * self._ep_size

    def _ranks_at_stage(self, stage: int) -> list[int]:
        """Return the global ranks assigned to a PP stage across all DP replicas."""
        inner = self._inner()
        modality_size = self._num_stages * inner
        result: list[int] = []
        for replica in range(self._dp_size):
            base = replica * modality_size + stage * inner
            result.extend(self._global_ranks[base : base + inner])
        return result

    def _my_flat_index_in_stage(self, stage: int) -> int:
        """Return the position of the current rank within its stage's rank list."""
        return self._ranks_at_stage(stage).index(self._my_rank)
