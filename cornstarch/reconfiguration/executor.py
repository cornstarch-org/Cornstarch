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

        Every rank participates in every collective call regardless of whether
        it currently holds a given parameter.  This avoids the deadlock that
        occurs in pipeline-parallel configurations when placeholder ranks skip
        ``dist.all_to_all`` while owner ranks call it.

        Two additional cases handled compared to the original implementation:

        * **Placeholder → owner** (rank gains a parameter): a new state-dict
          entry is created and keyed by the freshly-created ``nn.Parameter``
          (from :func:`set_param_by_name`).  For AMP optimizers that maintain
          separate master parameters the entry is keyed by the working parameter
          which is sufficient for all standard state-update operations.

        * **Owner → placeholder** (rank loses a parameter): the stale state
          entry is removed so it no longer consumes memory.
        """
        if optimizer is None:
            return

        try:
            master_to_working = optimizer.get_master_to_working_map()
        except AttributeError:
            master_to_working = None

        # Map working-param id → (master_param, state dict)
        wid_to_state: Dict[int, tuple] = {}
        for master_p, state in optimizer.optim.state.items():
            if master_to_working is not None:
                working = master_to_working.get(id(master_p))
                key = id(working) if working is not None else id(master_p)
            else:
                key = id(master_p)
            wid_to_state[key] = (master_p, state)

        # param_snapshot holds the parameter objects as they were *before*
        # execute() ran so optimizer state lookups by identity still work.
        if param_snapshot is not None:
            name_to_working: Dict[str, nn.Parameter] = param_snapshot
        else:
            name_to_working = dict(self.model.named_parameters())

        all_params = get_all_param_names(source_ownership, target_ownership)

        for param_name in sorted(all_params):
            is_target_owner = self._is_held_in_target(
                param_name, self.rank, target_ownership
            )

            # Retrieve local state (may be None for placeholder ranks).
            working_param = name_to_working.get(param_name)
            local_master_param = None
            local_state: Optional[Dict] = None
            if working_param is not None:
                entry = wid_to_state.get(id(working_param))
                if entry is not None:
                    local_master_param, local_state = entry

            # --- Step 1: ALL ranks agree on state keys AND their value types ---
            # Combining name + type in one all_gather_object guarantees that
            # every rank enters the same branches in Step 2, keeping every
            # subsequent collective (all_gather_object / all_to_all) symmetric.
            # 'tensor' = torch.Tensor,  'scalar' = non-tensor Python value.
            local_key_info: Dict[str, str] = {}
            if local_state is not None:
                for k, v in local_state.items():
                    local_key_info[k] = "tensor" if isinstance(v, torch.Tensor) else "scalar"
            all_key_info_list = [None] * self.world_size
            if dist.is_initialized():
                dist.all_gather_object(all_key_info_list, local_key_info)
            else:
                all_key_info_list[self.rank] = local_key_info

            canonical_key_info: Dict[str, str] = {}
            for kinfo in all_key_info_list:
                if kinfo:
                    canonical_key_info = kinfo
                    break

            if not canonical_key_info:
                # No rank holds optimizer state for this parameter.
                continue

            # --- Step 2: Redistribute each state value (all ranks participate) ---
            new_state: Dict = {}
            for state_key, key_kind in canonical_key_info.items():
                local_val = (
                    local_state.get(state_key) if local_state is not None else None
                )

                if key_kind == "scalar":
                    # Non-tensor scalar (e.g. integer step counter in old PyTorch).
                    # ALL ranks call all_gather_object so the collective is symmetric.
                    scalar_list = [None] * self.world_size
                    if dist.is_initialized():
                        dist.all_gather_object(scalar_list, local_val)
                    else:
                        scalar_list[self.rank] = local_val
                    canonical_val = next((v for v in scalar_list if v is not None), None)
                    new_state[state_key] = canonical_val
                    continue

                # Tensor state: every rank calls _redistribute_state_tensor so
                # that the underlying collective is symmetric.
                # Non-holding ranks pass None; _redistribute_state_tensor handles
                # the zero-tensor participation transparently.
                redistributed = self._redistribute_state_tensor(
                    param_name, local_val, source_ownership, target_ownership
                )

                if redistributed is not None:
                    new_state[state_key] = redistributed
                elif local_val is not None:
                    # This rank received nothing (not a target owner) but had a
                    # local value; keep it as a placeholder until cleaned up below.
                    new_state[state_key] = local_val

            # --- Step 3: Persist redistributed state for target owners only ---
            if not is_target_owner:
                # Remove stale state entries for parameters this rank no longer owns.
                if local_master_param is not None and local_master_param in optimizer.optim.state:
                    del optimizer.optim.state[local_master_param]
                continue

            if local_master_param is not None:
                # Rank was already an owner; update existing entry in-place.
                existing_state = optimizer.optim.state[local_master_param]
                existing_state.clear()
                existing_state.update(new_state)
            else:
                # Rank is a newly-acquired owner.  set_param_by_name created a
                # fresh nn.Parameter; use it as the state key.  For AMP
                # optimizers a proper master param would be needed, but the
                # working param key is sufficient for all practical state ops.
                try:
                    new_param = get_param_by_name(self.model, param_name)
                    optimizer.optim.state[new_param] = new_state
                except KeyError:
                    pass

    def _redistribute_state_tensor(
        self,
        param_name: str,
        state_tensor: Optional[torch.Tensor],
        source_ownership: Dict[int, LayerOwnership],
        target_ownership: Dict[int, LayerOwnership],
    ) -> Optional[torch.Tensor]:
        """Redistribute a single optimizer state tensor.

        ``state_tensor`` may be ``None`` when this rank does not currently hold
        the parameter (placeholder rank).  All ranks must call this method for
        the same ``param_name`` so that the underlying ``dist.all_to_all``
        collective is called by every participant.

        0-dim tensors (e.g. the ``step`` counter in Adam) are not sharded along
        any parameter axis.  They are collected from all ranks via
        ``all_gather_object`` and the first non-None value is returned.

        Returns the redistributed tensor for target-owner ranks, or ``None``
        for non-owner ranks (caller should ignore the return value in that case).
        """
        pieces = self._get_transfer_pieces(
            param_name, source_ownership, target_ownership
        )

        # Use _gather_metadata so every rank agrees on shape / dtype / device,
        # even when state_tensor is None on this rank.
        canonical_shape, canonical_dtype, target_device = self._gather_metadata(
            state_tensor, param_name=param_name
        )
        if canonical_shape is None:
            return None

        # 0-dim tensors (e.g. Adam's step counter) are global scalars, not
        # parameter shards.  Collect from all ranks and return the canonical value.
        if len(canonical_shape) == 0:
            scalar_val = state_tensor.item() if state_tensor is not None else None
            scalar_list = [None] * self.world_size
            if dist.is_initialized():
                dist.all_gather_object(scalar_list, scalar_val)
            else:
                scalar_list[self.rank] = scalar_val
            canonical_val = next((v for v in scalar_list if v is not None), None)
            if canonical_val is None:
                return None
            return torch.tensor(canonical_val, dtype=canonical_dtype, device=target_device)

        is_tp_sharded = bool(pieces) and pieces[0].shard_dim is not None
        shard_dim = pieces[0].shard_dim if pieces else None

        if is_tp_sharded:
            piece = pieces[0]
            piece_size = piece.src_local_end - piece.src_local_start
            xfer_shape = list(canonical_shape)
            xfer_shape[shard_dim] = piece_size
            xfer_shape = torch.Size(xfer_shape)
        else:
            xfer_shape = canonical_shape

        my_sends: Dict[int, TransferPiece] = {}
        my_recvs: Dict[int, TransferPiece] = {}
        for p in pieces:
            if p.src_rank == self.rank:
                my_sends[p.dst_rank] = p
            if p.dst_rank == self.rank:
                my_recvs[p.src_rank] = p

        input_tensor_list = []
        for dst_rank in range(self.world_size):
            if dst_rank in my_sends and state_tensor is not None:
                tp = my_sends[dst_rank]
                if tp.src_local_start is None:
                    chunk = state_tensor.data.contiguous()
                else:
                    idx = [slice(None)] * len(canonical_shape)
                    idx[shard_dim] = slice(tp.src_local_start, tp.src_local_end)
                    chunk = state_tensor.data[tuple(idx)].contiguous()
                input_tensor_list.append(chunk)
            else:
                input_tensor_list.append(
                    torch.zeros(xfer_shape, dtype=canonical_dtype, device=target_device)
                )

        output_tensor_list = [
            torch.zeros(xfer_shape, dtype=canonical_dtype, device=target_device)
            for _ in range(self.world_size)
        ]

        if dist.is_initialized():
            dist.all_to_all(output_tensor_list, input_tensor_list)
        else:
            for i, t in enumerate(input_tensor_list):
                output_tensor_list[i].copy_(t)

        if not my_recvs:
            return None

        if not is_tp_sharded:
            src_rank, _ = next(iter(my_recvs.items()))
            recv = output_tensor_list[src_rank]
            return recv.to(target_device) if recv.device != target_device else recv

        # Assemble TP shard pieces.
        my_tgt_range = target_ownership[self.rank].shard_range.get(param_name)
        tgt_shard_size = my_tgt_range[1] - my_tgt_range[0]
        tgt_shape = list(canonical_shape)
        tgt_shape[shard_dim] = tgt_shard_size
        assembled = torch.zeros(tgt_shape, dtype=canonical_dtype, device=target_device)
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
        """All-gather tensor shape / dtype / device so every rank has consistent info.

        Uses a single ``all_gather_object`` call (combining shape, dtype, and
        device) so that non-holding ranks can build zero tensors on the correct
        device for the all_to_all collective.  The canonical device is the first
        non-CPU device found across all ranks; this prevents non-holding ranks
        from accidentally creating CPU tensors that would break NCCL collectives.
        """
        if local_tensor is not None:
            local_shape: Optional[list] = list(local_tensor.shape)
            local_dtype: Optional[torch.dtype] = local_tensor.dtype
            local_device: Optional[str] = str(local_tensor.device)
        else:
            local_shape = None
            local_dtype = None
            try:
                _p = get_param_by_name(self.model, param_name)
                local_device = str(_p.device)
            except (KeyError, Exception):
                local_device = None

        if not dist.is_initialized():
            if local_shape is None:
                return None, None, local_device or "cpu"
            return torch.Size(local_shape), local_dtype, local_device or "cpu"

        info_list = [None] * self.world_size
        dist.all_gather_object(
            info_list,
            {"shape": local_shape, "dtype": local_dtype, "device": local_device},
        )

        canonical_shape: Optional[torch.Size] = None
        canonical_dtype: Optional[torch.dtype] = None
        canonical_device: str = local_device or "cpu"
        for info in info_list:
            if info is None:
                continue
            if canonical_shape is None and info.get("shape") is not None:
                canonical_shape = torch.Size(info["shape"])
                canonical_dtype = info["dtype"]
            dev = info.get("device")
            if dev is not None and "cpu" not in dev and "cpu" in canonical_device:
                canonical_device = dev

        return canonical_shape, canonical_dtype, canonical_device

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
