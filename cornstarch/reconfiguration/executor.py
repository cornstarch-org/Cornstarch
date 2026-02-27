"""Executor for dynamic parallel reconfiguration using all-to-all.

The core idea: instead of gathering TP shards to rank 0 and then
redistributing full tensors, each rank computes *exactly which slice of its
local shard* must go to which destination rank, and sends only that slice.
This works for any combination of source/target TP sizes and collapses the
old three-phase (unshards → redistribute → reshards) into a single all-to-all.

For DP redundancy across TP transfers: when multiple source ranks hold the
*same* range (DP replicas), the piece-assignment round-robins across them to
spread the sending load instead of leaving replicas idle.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from cornstarch.shardformer.shard.placeholder import TensorPlaceholder

from .data_structures import LayerOwnership
from .utils import get_all_param_names, get_param_by_name, set_param_by_name


class TransferPiece(NamedTuple):
    """One directed tensor transfer from src_rank to dst_rank.

    For TP-sharded parameters the piece is a contiguous slice of the local
    shard along ``shard_dim``.  For non-TP parameters all slice fields are
    ``None`` meaning "transfer the whole tensor".
    """
    src_rank: int
    dst_rank: int
    src_local_start: Optional[int]
    src_local_end: Optional[int]
    dst_local_start: Optional[int]
    dst_local_end: Optional[int]
    shard_dim: Optional[int]


class ReconfigurationExecutor:
    """Executes tensor redistribution using torch.distributed.all_to_all."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
    ) -> None:
        """Redistribute every parameter from source to target ownership."""
        all_params = get_all_param_names(source_ownership, target_ownership)
        for param_name in sorted(all_params):
            self._redistribute_param(param_name, source_ownership, target_ownership)

    # ------------------------------------------------------------------
    # Transfer-piece computation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transfer_pieces(
        param_name: str,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
    ) -> List[TransferPiece]:
        """Compute exact (src_rank, dst_rank, slice) triples for *param_name*.

        Algorithm
        ---------
        1.  Collect all shard-range *breakpoints* from source and target
            ownership.  Consecutive breakpoints define equal-sized "pieces".
        2.  For each piece [start:end]:
            - **src_holders** = source ranks whose local shard covers [start:end].
              (Multiple ranks cover the same piece when they are DP replicas.)
            - **dst_needers**  = target ranks whose required shard covers [start:end].
            - Assignment: each dst_needer is paired with the src_holder that has
              sent the fewest pieces so far (load-balanced round-robin).  This
              uses *all* DP source replicas rather than leaving them idle.
        3.  For purely non-TP parameters (all shard_ranges are ``None``): skip
            the breakpoint logic and fall back to the classic round-robin over
            whole tensors.

        Returns a list of :class:`TransferPiece` objects; may be empty when the
        parameter does not exist in either ownership.
        """
        # Collect per-rank shard ranges for this param (skip placeholders).
        src_ranges: Dict[int, Optional[Tuple[int, int]]] = {}
        tgt_ranges: Dict[int, Optional[Tuple[int, int]]] = {}
        src_shard_dim: Optional[int] = None
        tgt_shard_dim: Optional[int] = None

        for r, own in source_ownership.items():
            if own.is_placeholder.get(param_name, True):
                continue
            if param_name not in own.layer_names:
                continue
            sr = own.shard_range.get(param_name)
            src_ranges[r] = sr
            if sr is not None and src_shard_dim is None:
                src_shard_dim = own.shard_dim.get(param_name)

        for r, own in target_ownership.items():
            if own.is_placeholder.get(param_name, True):
                continue
            if param_name not in own.layer_names:
                continue
            tr = own.shard_range.get(param_name)
            tgt_ranges[r] = tr
            if tr is not None and tgt_shard_dim is None:
                tgt_shard_dim = own.shard_dim.get(param_name)

        if not src_ranges or not tgt_ranges:
            return []

        shard_dim = src_shard_dim if src_shard_dim is not None else tgt_shard_dim
        all_src_none = all(v is None for v in src_ranges.values())
        all_tgt_none = all(v is None for v in tgt_ranges.values())

        # --- Non-TP case: whole-tensor round-robin ---
        if all_src_none and all_tgt_none:
            src_candidates = sorted(src_ranges.keys())
            dst_candidates = sorted(tgt_ranges.keys())
            return [
                TransferPiece(
                    src_rank=src_candidates[i % len(src_candidates)],
                    dst_rank=dst,
                    src_local_start=None,
                    src_local_end=None,
                    dst_local_start=None,
                    dst_local_end=None,
                    shard_dim=None,
                )
                for i, dst in enumerate(dst_candidates)
            ]

        # --- TP case: breakpoint-based piece assignment ---
        full_size = max(
            max((r[1] for r in src_ranges.values() if r is not None), default=0),
            max((r[1] for r in tgt_ranges.values() if r is not None), default=0),
        )
        if full_size == 0:
            return []

        breakpoints = {0, full_size}
        for r in src_ranges.values():
            if r is not None:
                breakpoints.update(r)
        for r in tgt_ranges.values():
            if r is not None:
                breakpoints.update(r)
        bp_sorted = sorted(breakpoints)

        # send_count[src_rank] tracks how many pieces this source has been
        # assigned so far, used for load-balanced selection.
        send_count: Dict[int, int] = {r: 0 for r in src_ranges}

        pieces: List[TransferPiece] = []
        for i in range(len(bp_sorted) - 1):
            piece_start, piece_end = bp_sorted[i], bp_sorted[i + 1]

            # Source ranks whose shard fully contains this piece.
            src_holders = sorted(
                r
                for r, sr in src_ranges.items()
                if sr is None or (sr[0] <= piece_start and sr[1] >= piece_end)
            )
            # Target ranks whose required shard fully contains this piece.
            dst_needers = sorted(
                r
                for r, tr in tgt_ranges.items()
                if tr is None or (tr[0] <= piece_start and tr[1] >= piece_end)
            )

            if not src_holders or not dst_needers:
                continue

            for dst in dst_needers:
                # Pick the source holder with the fewest assignments so far.
                src = min(src_holders, key=lambda r: send_count[r])
                send_count[src] += 1

                src_sr = src_ranges[src]
                dst_tr = tgt_ranges[dst]
                src_offset = src_sr[0] if src_sr is not None else 0
                dst_offset = dst_tr[0] if dst_tr is not None else 0

                pieces.append(
                    TransferPiece(
                        src_rank=src,
                        dst_rank=dst,
                        src_local_start=piece_start - src_offset,
                        src_local_end=piece_end - src_offset,
                        dst_local_start=piece_start - dst_offset,
                        dst_local_end=piece_end - dst_offset,
                        shard_dim=shard_dim,
                    )
                )

        return pieces

    # ------------------------------------------------------------------
    # Parameter redistribution
    # ------------------------------------------------------------------

    def _redistribute_param(
        self,
        param_name: str,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
    ) -> None:
        """Redistribute a single parameter using all-to-all.

        For each parameter, ``_get_transfer_pieces`` tells us the exact slice
        each source rank must send to each destination rank.  The all-to-all
        uses tensors of size *piece_size* (rather than the full-param size),
        so no intermediate full-tensor assembly is ever needed.

        After the all-to-all, destination ranks that received multiple pieces
        (e.g. because the target TP shard spans several source TP shards)
        assemble them in-place into the final shard-shaped tensor.
        """
        pieces = self._get_transfer_pieces(
            param_name, source_ownership, target_ownership
        )
        if not pieces:
            if not self._is_held_in_target(param_name, self.rank, target_ownership):
                self._replace_with_placeholder(param_name)
            return

        is_tp_sharded = pieces[0].shard_dim is not None

        # --- Gather per-param metadata (shape, dtype) via all_gather ---
        local_tensor: Optional[torch.Tensor] = None
        if self._is_held_in_source(param_name, self.rank, source_ownership):
            try:
                local_tensor = get_param_by_name(self.model, param_name)
            except KeyError:
                pass

        canonical_shape, canonical_dtype, target_device = self._gather_metadata(
            local_tensor, param_name=param_name
        )
        if canonical_shape is None:
            return

        # --- Determine per-all_to_all tensor shape (piece vs. full) ---
        shard_dim = pieces[0].shard_dim
        if is_tp_sharded:
            # All pieces are uniform in size (standard uniform TP sharding).
            piece = pieces[0]
            piece_size = piece.src_local_end - piece.src_local_start
            xfer_shape = list(canonical_shape)
            xfer_shape[shard_dim] = piece_size
            xfer_shape = torch.Size(xfer_shape)
        else:
            xfer_shape = canonical_shape

        # Index pieces by my role as sender / receiver.
        my_sends: Dict[int, TransferPiece] = {}
        my_recvs: Dict[int, TransferPiece] = {}
        for p in pieces:
            if p.src_rank == self.rank:
                my_sends[p.dst_rank] = p
            if p.dst_rank == self.rank:
                my_recvs[p.src_rank] = p

        # --- Build all_to_all input ---
        input_tensor_list = []
        for dst_rank in range(self.world_size):
            if dst_rank in my_sends:
                tp = my_sends[dst_rank]
                if tp.src_local_start is None:
                    chunk = local_tensor.data.contiguous()
                else:
                    idx = [slice(None)] * len(canonical_shape)
                    idx[shard_dim] = slice(tp.src_local_start, tp.src_local_end)
                    chunk = local_tensor.data[tuple(idx)].contiguous()
                input_tensor_list.append(chunk)
            else:
                input_tensor_list.append(
                    torch.zeros(xfer_shape, dtype=canonical_dtype, device=target_device)
                )

        # --- Build all_to_all output buffers ---
        output_tensor_list = [
            torch.zeros(xfer_shape, dtype=canonical_dtype, device=target_device)
            for _ in range(self.world_size)
        ]

        # --- Execute ---
        if dist.is_initialized():
            dist.all_to_all(output_tensor_list, input_tensor_list)
        else:
            for i, t in enumerate(input_tensor_list):
                output_tensor_list[i].copy_(t)

        # --- Assemble received pieces into the target shard ---
        if my_recvs:
            if not is_tp_sharded:
                # Single whole-tensor transfer.
                src_rank, _ = next(iter(my_recvs.items()))
                recv = output_tensor_list[src_rank]
                if recv.device != target_device:
                    recv = recv.to(target_device)
                set_param_by_name(self.model, param_name, recv)
            else:
                # Compute target shard shape from the pieces this rank receives.
                my_tgt_range = target_ownership[self.rank].shard_range.get(param_name)
                tgt_shard_size = my_tgt_range[1] - my_tgt_range[0]
                tgt_shape = list(canonical_shape)
                tgt_shape[shard_dim] = tgt_shard_size
                assembled = torch.zeros(
                    tgt_shape, dtype=canonical_dtype, device=target_device
                )
                for src_rank, tp in my_recvs.items():
                    piece = output_tensor_list[src_rank]
                    if piece.device != target_device:
                        piece = piece.to(target_device)
                    idx = [slice(None)] * len(tgt_shape)
                    idx[shard_dim] = slice(tp.dst_local_start, tp.dst_local_end)
                    assembled[tuple(idx)] = piece
                set_param_by_name(self.model, param_name, assembled)

        if not self._is_held_in_target(param_name, self.rank, target_ownership):
            self._replace_with_placeholder(param_name)

    # ------------------------------------------------------------------
    # Optimizer state redistribution
    # ------------------------------------------------------------------

    def redistribute_optimizer_states(
        self,
        optimizer,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
        param_snapshot: Optional[Dict[str, "nn.Parameter"]] = None,
    ) -> None:
        """Redistribute optimizer states to match the new parameter ownership.

        Mirrors :meth:`execute` but operates on the optimizer's internal state
        tensors (e.g. ``exp_avg``, ``exp_avg_sq``).  Each state tensor for a
        parameter is moved with the same piece-based all-to-all pattern as the
        parameter itself.
        """
        if optimizer is None:
            return

        try:
            master_to_working = optimizer.get_master_to_working_map()
        except AttributeError:
            master_to_working = None

        wid_to_state: Dict[int, tuple] = {}
        for master_p, state in optimizer.optim.state.items():
            if master_to_working is not None:
                working = master_to_working.get(id(master_p))
                key = id(working) if working is not None else id(master_p)
            else:
                key = id(master_p)
            wid_to_state[key] = (master_p, state)

        if param_snapshot is not None:
            name_to_working: Dict[str, nn.Parameter] = param_snapshot
        else:
            name_to_working = dict(self.model.named_parameters())

        all_params = get_all_param_names(source_ownership, target_ownership)

        for param_name in sorted(all_params):
            working_param = name_to_working.get(param_name)
            if working_param is None:
                continue
            entry = wid_to_state.get(id(working_param))
            if entry is None:
                continue
            _, state = entry

            is_target_owner = self._is_held_in_target(
                param_name, self.rank, target_ownership
            )
            new_state: Dict = {}
            for state_key, state_val in state.items():
                if not isinstance(state_val, torch.Tensor):
                    new_state[state_key] = state_val
                    continue
                redistributed = self._redistribute_state_tensor(
                    param_name, state_val, source_ownership, target_ownership
                )
                # Only adopt the redistributed tensor if this rank is a target
                # owner.  Non-owner ranks still participate in the all_to_all
                # collective (required) but must not corrupt their stale state.
                new_state[state_key] = redistributed if is_target_owner else state_val
            state.clear()
            state.update(new_state)

    def _redistribute_state_tensor(
        self,
        param_name: str,
        state_tensor: torch.Tensor,
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
    ) -> torch.Tensor:
        """Redistribute a single optimizer state tensor.

        Uses the same piece-based logic as :meth:`_redistribute_param` since
        optimizer state tensors are sharded identically to their parameters.
        """
        pieces = self._get_transfer_pieces(
            param_name, source_ownership, target_ownership
        )

        tensor_shape = state_tensor.shape
        tensor_dtype = state_tensor.dtype
        target_device = state_tensor.device

        is_tp_sharded = bool(pieces) and pieces[0].shard_dim is not None
        shard_dim = pieces[0].shard_dim if pieces else None

        if is_tp_sharded:
            piece = pieces[0]
            piece_size = piece.src_local_end - piece.src_local_start
            xfer_shape = list(tensor_shape)
            xfer_shape[shard_dim] = piece_size
            xfer_shape = torch.Size(xfer_shape)
        else:
            xfer_shape = tensor_shape

        my_sends: Dict[int, TransferPiece] = {}
        my_recvs: Dict[int, TransferPiece] = {}
        for p in pieces:
            if p.src_rank == self.rank:
                my_sends[p.dst_rank] = p
            if p.dst_rank == self.rank:
                my_recvs[p.src_rank] = p

        input_tensor_list = []
        for dst_rank in range(self.world_size):
            if dst_rank in my_sends:
                tp = my_sends[dst_rank]
                if tp.src_local_start is None:
                    chunk = state_tensor.data.contiguous()
                else:
                    idx = [slice(None)] * len(tensor_shape)
                    idx[shard_dim] = slice(tp.src_local_start, tp.src_local_end)
                    chunk = state_tensor.data[tuple(idx)].contiguous()
                input_tensor_list.append(chunk)
            else:
                input_tensor_list.append(
                    torch.zeros(xfer_shape, dtype=tensor_dtype, device=target_device)
                )

        output_tensor_list = [
            torch.zeros(xfer_shape, dtype=tensor_dtype, device=target_device)
            for _ in range(self.world_size)
        ]

        if dist.is_initialized():
            dist.all_to_all(output_tensor_list, input_tensor_list)
        else:
            for i, t in enumerate(input_tensor_list):
                output_tensor_list[i].copy_(t)

        if not my_recvs:
            return torch.zeros(tensor_shape, dtype=tensor_dtype, device=target_device)

        if not is_tp_sharded:
            src_rank, _ = next(iter(my_recvs.items()))
            recv = output_tensor_list[src_rank]
            return recv.to(target_device) if recv.device != target_device else recv

        # Assemble TP shard pieces.
        my_tgt_range = target_ownership[self.rank].shard_range.get(param_name)
        tgt_shard_size = my_tgt_range[1] - my_tgt_range[0]
        tgt_shape = list(tensor_shape)
        tgt_shape[shard_dim] = tgt_shard_size
        assembled = torch.zeros(tgt_shape, dtype=tensor_dtype, device=target_device)
        for src_rank, tp in my_recvs.items():
            piece = output_tensor_list[src_rank]
            if piece.device != target_device:
                piece = piece.to(target_device)
            idx = [slice(None)] * len(tgt_shape)
            idx[shard_dim] = slice(tp.dst_local_start, tp.dst_local_end)
            assembled[tuple(idx)] = piece
        return assembled

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gather_metadata(
        self,
        local_tensor: Optional[torch.Tensor],
        param_name: str = "",
    ) -> Tuple[Optional[torch.Size], Optional[torch.dtype], str]:
        """All-gather tensor shape / dtype so every rank has consistent info.

        ``target_device`` is resolved from whichever tensor is available on
        this rank — either the source tensor or whatever copy of the parameter
        still lives in the model.  We never blindly fall back to the current
        CUDA device; if no tensor is accessible, we default to CPU.
        """
        local_shape = list(local_tensor.shape) if local_tensor is not None else None
        local_dtype = local_tensor.dtype if local_tensor is not None else None

        # Determine the device from the local tensor (preferred) or from
        # whatever copy of the param still resides in the model.
        if local_tensor is not None:
            target_device = local_tensor.device
        else:
            try:
                _p = get_param_by_name(self.model, param_name)
                target_device = _p.device
            except (KeyError, Exception):
                target_device = "cpu"

        if not dist.is_initialized():
            if local_shape is None:
                return None, None, target_device
            return torch.Size(local_shape), local_dtype, target_device

        shape_list = [None] * self.world_size
        dtype_list = [None] * self.world_size
        dist.all_gather_object(shape_list, local_shape)
        dist.all_gather_object(dtype_list, local_dtype)

        canonical_shape = None
        canonical_dtype = None
        for s, d in zip(shape_list, dtype_list):
            if s is not None and d is not None:
                canonical_shape = torch.Size(s)
                canonical_dtype = d
                break

        return canonical_shape, canonical_dtype, target_device

    def _is_held_in_source(
        self,
        param_name: str,
        rank: int,
        source_ownership: Dict[int, LayerOwnership],
    ) -> bool:
        return (
            rank in source_ownership
            and param_name in source_ownership[rank].layer_names
            and not source_ownership[rank].is_placeholder.get(param_name, False)
        )

    def _is_held_in_target(
        self,
        param_name: str,
        rank: int,
        target_ownership: Dict[int, LayerOwnership],
    ) -> bool:
        return (
            rank in target_ownership
            and param_name in target_ownership[rank].layer_names
            and not target_ownership[rank].is_placeholder.get(param_name, False)
        )

    def _replace_with_placeholder(self, param_name: str) -> None:
        """Replace a model parameter with a TensorPlaceholder."""
        try:
            param = get_param_by_name(self.model, param_name)
            placeholder = TensorPlaceholder(param)

            parts = param_name.split(".")
            module = self.model
            for part in parts[:-1]:
                module = module[int(part)] if part.isdigit() else getattr(module, part)

            if not hasattr(module, "_parameter_placeholders"):
                module._parameter_placeholders = {}
            module._parameter_placeholders[parts[-1]] = placeholder

            delattr(module, parts[-1])
            setattr(module, parts[-1], None)
        except KeyError:
            pass
