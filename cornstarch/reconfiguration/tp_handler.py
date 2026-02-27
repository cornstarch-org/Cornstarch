"""TP un-shard and re-shard handler for reconfiguration."""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class TPReconfigurationHandler:
    """Handles TP un-sharding and re-sharding during parallel reconfiguration.

    Typical usage::

        handler = TPReconfigurationHandler()
        handler.unshards(model, old_tp_group, optimizer)
        # ... redistribute params via ReconfigurationExecutor ...
        handler.reshards(model, new_tp_group, optimizer)

    After ``unshards``:
    - TP rank 0 holds the full (gathered) parameter tensor.
    - All other TP ranks hold a zero-filled tensor of the same full shape so
      that ``TensorOwnershipAnalyzer`` / ``ReconfigurationExecutor`` can treat
      them as non-owners without special casing.

    After ``reshards``:
    - Every rank in the new TP group holds the correct contiguous slice of the
      parameter according to its new TP rank.
    """

    def __init__(self) -> None:
        # (full_param_name, shard_dim) for every param that was TP-sharded,
        # recorded during unshards() and consumed by reshards().
        self._shard_specs: List[Tuple[str, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def unshards(
        self,
        model: nn.Module,
        tp_group: Optional[dist.ProcessGroup],
        optimizer=None,
    ) -> None:
        """All-gather TP-sharded parameters to full tensors (no-op when TP=1).

        Args:
            model: The parallelized model (may contain Linear1D_Col/Row).
            tp_group: Current TP process group.  ``None`` or size-1 → no-op.
            optimizer: Optional ColossalAI ``OptimizerWrapper``.  When given,
                the master params and optimizer state tensors that correspond
                to TP-sharded working params are gathered the same way.
        """
        if tp_group is None or dist.get_world_size(tp_group) == 1:
            self._shard_specs = []
            return

        try:
            from colossalai.shardformer.layer import Linear1D_Col, Linear1D_Row
            _tp_classes = (Linear1D_Col, Linear1D_Row)
        except ImportError:
            self._shard_specs = []
            return

        tp_rank = dist.get_rank(tp_group)
        self._shard_specs = []

        # Build working-param → (master-param, state) mapping for optimizer.
        _working_ptr_to_optim = self._build_optim_map(optimizer)

        for module_name, module in model.named_modules():
            if not isinstance(module, _tp_classes):
                continue

            is_col = isinstance(module, Linear1D_Col)
            # Linear1D_Col shards dim-0 (out_features); Linear1D_Row shards dim-1.
            weight_shard_dim = 0 if is_col else 1

            for local_name, param in list(module.named_parameters(recurse=False)):
                if param is None:
                    continue

                if local_name == "weight":
                    shard_dim = weight_shard_dim
                elif local_name == "bias" and is_col:
                    shard_dim = 0
                else:
                    continue

                full_param_name = (
                    f"{module_name}.{local_name}" if module_name else local_name
                )

                # Gather shards to TP rank 0 only (not all_gather).  Non-zero
                # ranks send their shard and receive nothing; only rank 0 builds
                # the full tensor.  All other ranks receive a zero-filled tensor
                # of the full shape so the executor treats them as non-owners.
                full = self._gather_dim(param.data, shard_dim, tp_group)

                if tp_rank == 0:
                    param.data = full.to(param.device)
                    self._shard_specs.append((full_param_name, shard_dim))
                else:
                    # Infer full shape from own shard (all shards have equal size).
                    full_shape = list(param.data.shape)
                    full_shape[shard_dim] *= dist.get_world_size(tp_group)
                    param.data = torch.zeros(
                        full_shape, dtype=param.dtype, device=param.device
                    )

                # Gather optimizer states for this parameter.
                self._gather_optim_state(
                    param, shard_dim, tp_group, tp_rank, _working_ptr_to_optim, full
                )

    def reshards(
        self,
        model: nn.Module,
        new_tp_group: Optional[dist.ProcessGroup],
        optimizer=None,
    ) -> None:
        """Scatter full parameters from new TP rank 0 to all new TP ranks.

        Uses the shard specs recorded by :meth:`unshards`.  After the PP-level
        redistribution performed by ``ReconfigurationExecutor``, new TP rank 0
        holds the full parameter tensors.  This method uses ``dist.scatter`` so
        that each rank receives only its own chunk: TP rank 0 sends exactly
        ``chunk_size`` bytes to each peer rather than broadcasting the full
        tensor and having every rank slice locally.

        Args:
            model: Model whose parameters should be re-sharded.
            new_tp_group: New TP process group (after reconfiguration).
            optimizer: Optional ``OptimizerWrapper``.
        """
        if new_tp_group is None or dist.get_world_size(new_tp_group) == 1:
            return
        if not self._shard_specs:
            return

        new_tp_rank = dist.get_rank(new_tp_group)
        new_tp_size = dist.get_world_size(new_tp_group)
        global_src = dist.get_global_rank(new_tp_group, 0)

        param_map = dict(model.named_parameters())
        _working_ptr_to_optim = self._build_optim_map(optimizer)

        for full_param_name, shard_dim in self._shard_specs:
            param = param_map.get(full_param_name)
            if param is None:
                continue

            total = param.data.shape[shard_dim]
            if total % new_tp_size != 0:
                raise ValueError(
                    f"Parameter '{full_param_name}' dim {shard_dim} size {total} "
                    f"is not divisible by new TP size {new_tp_size}."
                )
            chunk = total // new_tp_size

            # Build chunk shape for the receive buffer.
            recv_shape = list(param.data.shape)
            recv_shape[shard_dim] = chunk
            recv = torch.empty(recv_shape, dtype=param.dtype, device=param.device)

            if new_tp_rank == 0:
                # Slice the full tensor into per-rank chunks.
                scatter_list = []
                for i in range(new_tp_size):
                    slices = [slice(None)] * param.data.dim()
                    slices[shard_dim] = slice(i * chunk, (i + 1) * chunk)
                    scatter_list.append(
                        param.data[tuple(slices)].contiguous().to(param.device)
                    )
                dist.scatter(recv, scatter_list=scatter_list,
                             src=global_src, group=new_tp_group)
            else:
                dist.scatter(recv, scatter_list=None,
                             src=global_src, group=new_tp_group)

            param.data = recv

            # Scatter optimizer states the same way.
            self._scatter_optim_state(
                param, shard_dim, new_tp_group, new_tp_rank, new_tp_size,
                global_src, chunk, _working_ptr_to_optim,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_dim(
        tensor: torch.Tensor,
        dim: int,
        group: dist.ProcessGroup,
    ) -> Optional[torch.Tensor]:
        """Gather shards from all TP ranks to TP rank 0 and concatenate.

        Unlike ``all_gather``, only the destination (TP rank 0) allocates and
        receives the full tensor.  All other ranks merely send their shard and
        receive nothing, halving their inbound bandwidth usage.

        Returns:
            The full concatenated tensor on TP rank 0; ``None`` on all other
            ranks (they have no use for the full tensor until ``reshards()``).
        """
        tp_size = dist.get_world_size(group)
        tp_rank = dist.get_rank(group)
        global_dst = dist.get_global_rank(group, 0)

        if tp_rank == 0:
            gather_list = [torch.empty_like(tensor) for _ in range(tp_size)]
        else:
            gather_list = None

        dist.gather(tensor.contiguous(), gather_list=gather_list,
                    dst=global_dst, group=group)

        if tp_rank == 0:
            return torch.cat(gather_list, dim=dim)
        return None

    @staticmethod
    def _build_optim_map(optimizer) -> dict:
        """Return {working_param_data_ptr: (master_param, state_dict)} map."""
        if optimizer is None:
            return {}
        result = {}
        try:
            master_to_working = optimizer.get_master_to_working_map()
        except AttributeError:
            master_to_working = None

        for master_p, state in optimizer.optim.state.items():
            if master_to_working is not None:
                working = master_to_working.get(id(master_p))
                key = working.data_ptr() if working is not None else master_p.data_ptr()
            else:
                key = master_p.data_ptr()
            result[key] = (master_p, state)
        return result

    @staticmethod
    def _gather_optim_state(
        working_param, shard_dim, tp_group, tp_rank, ptr_map, _unused=None
    ):
        """Gather optimizer state shards to TP rank 0 using ``dist.gather``."""
        entry = ptr_map.get(working_param.data_ptr())
        if entry is None:
            return
        master_p, state = entry
        global_dst = dist.get_global_rank(tp_group, 0)
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.shape == working_param.shape:
                full_v = TPReconfigurationHandler._gather_dim(
                    v.contiguous(), shard_dim, tp_group
                )
                if tp_rank == 0:
                    new_state[k] = full_v.to(v.device)
                else:
                    full_shape = list(v.shape)
                    full_shape[shard_dim] *= dist.get_world_size(tp_group)
                    new_state[k] = torch.zeros(full_shape, dtype=v.dtype, device=v.device)
            else:
                new_state[k] = v
        master_p.grad = None
        state.clear()
        state.update(new_state)

    @staticmethod
    def _scatter_optim_state(
        working_param, shard_dim, new_tp_group, new_tp_rank, new_tp_size,
        global_src, chunk, ptr_map,
    ):
        """Scatter optimizer states using ``dist.scatter`` after PP redistribution."""
        entry = ptr_map.get(working_param.data_ptr())
        if entry is None:
            return
        master_p, state = entry
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 \
                    and v.shape[shard_dim] == new_tp_size * chunk:
                recv_shape = list(v.shape)
                recv_shape[shard_dim] = chunk
                recv = torch.empty(recv_shape, dtype=v.dtype, device=v.device)
                if new_tp_rank == 0:
                    scatter_list = []
                    for i in range(new_tp_size):
                        slices = [slice(None)] * v.dim()
                        slices[shard_dim] = slice(i * chunk, (i + 1) * chunk)
                        scatter_list.append(v[tuple(slices)].contiguous())
                    dist.scatter(recv, scatter_list=scatter_list,
                                 src=global_src, group=new_tp_group)
                else:
                    dist.scatter(recv, scatter_list=None,
                                 src=global_src, group=new_tp_group)
                new_state[k] = recv
            else:
                new_state[k] = v
        state.clear()
        state.update(new_state)
