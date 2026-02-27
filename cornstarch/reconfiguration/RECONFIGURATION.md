# Dynamic Parallel Reconfiguration

This module redistributes model parameters and optimizer states when the parallel
configuration changes (e.g. PP=1→2, TP=4→2, DP=4→1).  The entry point is
`MultimodalParallelPlugin.reconfigure()`.

---

## Core Insight — Direct Shard-to-Shard Transfer

The naive approach gathers each TP-sharded parameter to a single rank, redistributes
the full tensor, then scatters again.  This is unnecessary:

- Going **TP=4 → TP=2**: source rank 0 holds rows `[0:N/4]`, rank 1 holds `[N/4:N/2]`.
  Target rank 0 needs `[0:N/2]`.  Ranks 0 and 1 can send their shards **directly** to
  target rank 0 — no intermediate full-tensor assembly on any rank.
- **DP replicas** hold identical copies of each TP shard.  Instead of picking one
  replica arbitrarily and leaving the rest idle, the piece-assignment algorithm uses
  all replicas as load-balanced senders.

The key enabler is extending `LayerOwnership` to record not just *"does rank R own
param P?"* but *"which contiguous range of P does rank R hold?"*.

---

## Data Structures

### `LayerOwnership`

```
rank          : int
layer_names   : list[str]              # params held (fully or partially)
is_placeholder: dict[str, bool]        # True → TensorPlaceholder
shard_range   : dict[str, (int,int)?]  # [start, end) in full tensor, or None
shard_dim     : dict[str, int?]        # axis that is sharded, or None
```

`shard_range[p] = None` means this rank holds the complete parameter (no TP
sharding, or TP=1).

### `TransferPiece`

```
src_rank        : int
dst_rank        : int
src_local_start : int?   # slice in src's local tensor; None = whole tensor
src_local_end   : int?
dst_local_start : int?   # where in dst's local tensor to place it; None = whole
dst_local_end   : int?
shard_dim       : int?   # which axis; None = no slicing
```

---

## Algorithm

### Phase 1 — Analyze source ownership

`TensorOwnershipAnalyzer(model).analyze(tp_group=current_tp_group)`

For each parameter:
- If its parent module is `Linear1D_Col` (ColossalAI column-parallel linear):
  `shard_dim = 0`, `shard_range = (tp_rank * local_rows, (tp_rank+1) * local_rows)`
- If its parent module is `Linear1D_Row` (row-parallel):
  `shard_dim = 1`, `shard_range = (tp_rank * local_cols, (tp_rank+1) * local_cols)`
- Otherwise: `shard_range = None`, `shard_dim = None`

Result is all-gathered over the world group so every rank has the complete map.

### Phase 2 — Build new pg_mesh

Constructs `MultiModalProcessGroupMesh` from the new plugin configs (topology only —
no `dist.new_group` calls yet).  Group cache is seeded from the old mesh so unchanged
rank-set groups are reused.

### Phase 3 — Compute target ownership

`build_target_ownership(model, new_pg_mesh, ..., source_ownership=source_ownership)`

- Extracts `shard_dim` and `full_size` for each TP-sharded param from
  `source_ownership` (no additional communication needed).
- Iterates over **all** TP ranks in the new mesh (not just rank 0).
- For each (PP stage, DP replica, TP rank): records ownership with target shard range
  `(tp_idx * chunk, (tp_idx+1) * chunk)` where `chunk = full_size // new_tp_size`.
- Result is all-gathered over the world group.

### Phase 4 — Direct shard-to-shard redistribution

`ReconfigurationExecutor.execute(source_ownership, target_ownership)`

Calls `_redistribute_param` for each parameter in sorted order.

#### `_get_transfer_pieces(param_name, source_ownership, target_ownership)`

**Non-TP params** (`shard_range` is `None` for all ranks):

```
src_candidates = sorted source owners
dst_candidates = sorted target owners
pieces = [(src_candidates[i % len(src)], dst, None, None, None)]
         for i, dst in enumerate(dst_candidates)
```

Standard round-robin; each destination receives one complete tensor from one source.

**TP-sharded params**:

1. Collect all `start` and `end` values from source and target `shard_range`s, plus
   `{0, full_size}`.  Sort them → breakpoints `[bp₀, bp₁, bp₂, ...]`.

2. For each consecutive pair `[bpᵢ, bpᵢ₊₁]` ("piece"):

   ```
   src_holders = source ranks whose shard fully contains [bpᵢ, bpᵢ₊₁]
                 (includes all DP replicas covering that range)
   dst_needers = target ranks whose required shard contains [bpᵢ, bpᵢ₊₁]

   for each dst in dst_needers:
       src = src_holder with fewest transfers assigned so far   ← load-balanced
       emit TransferPiece(
           src, dst,
           src_local_start = bpᵢ   - src_shard_start,
           src_local_end   = bpᵢ₊₁ - src_shard_start,
           dst_local_start = bpᵢ   - dst_shard_start,
           dst_local_end   = bpᵢ₊₁ - dst_shard_start,
           shard_dim,
       )
   ```

**Uniform piece size guarantee**: with standard uniform TP sharding, all pieces have
size `full_size / lcm(src_tp_size, tgt_tp_size)`.  This means every position in the
all-to-all uses the same tensor shape `piece_shape`, satisfying the Gloo and NCCL
requirement that `input_tensor_list[i]` has the same shape across all ranks for a
given `i`.

#### Example — TP=4 → TP=2

Full tensor rows: `[0, N/4, N/2, 3N/4, N]`.  Breakpoints: `{0, N/4, N/2, 3N/4, N}`.
`lcm(4, 2) = 4` → piece size = `N/4`.

| Piece | src_holders | dst_needers | Assigned transfer |
|-------|-------------|-------------|-------------------|
| [0 : N/4]   | rank 0 | rank 0 | rank 0 → rank 0, local [0:N/4] → dst [0:N/4] |
| [N/4 : N/2] | rank 1 | rank 0 | rank 1 → rank 0, local [0:N/4] → dst [N/4:N/2] |
| [N/2 : 3N/4]| rank 2 | rank 1 | rank 2 → rank 1, local [0:N/4] → dst [0:N/4] |
| [3N/4 : N]  | rank 3 | rank 1 | rank 3 → rank 1, local [0:N/4] → dst [N/4:N/2] |

No gather. No idle ranks.

#### Example — TP=2, DP=2 (source) → TP=2, DP=1 (target)

Source: ranks 0,2 both hold `[0:N/2]` (DP replicas); ranks 1,3 both hold `[N/2:N]`.

| Piece | src_holders | dst_needers | Assigned |
|-------|-------------|-------------|----------|
| [0 : N/2] | rank 0, rank 2 | dst 0 | rank 0 (fewer sends) |
| [N/2 : N] | rank 1, rank 3 | dst 1 | rank 1 (fewer sends) |

Ranks 2 and 3 are idle here because there is only one destination per range.  If the
target also had DP=2 (two destinations per range), all four source ranks would be
utilised.

#### `_redistribute_param` — all-to-all execution

```python
# Build input_tensor_list (piece_shape at every position)
for dst in range(world_size):
    if dst in my_sends:
        tp = my_sends[dst]
        chunk = local_tensor[..., tp.src_local_start:tp.src_local_end, ...]  # along shard_dim
    else:
        chunk = zeros(piece_shape)
    input_tensor_list[dst] = chunk

output_tensor_list = [zeros(piece_shape)] * world_size

dist.all_to_all(output_tensor_list, input_tensor_list)

# Assemble received pieces into target shard
if my_recvs:
    assembled = zeros(target_shard_shape)
    for src, tp in my_recvs.items():
        assembled[..., tp.dst_local_start:tp.dst_local_end, ...] = output_tensor_list[src]
    set_param_by_name(model, param_name, assembled)
```

### Phase 5 — Redistribute optimizer states

`ReconfigurationExecutor.redistribute_optimizer_states(optimizer, source_ownership, target_ownership)`

Same piece-based all-to-all pattern applied to each state tensor (`exp_avg`,
`exp_avg_sq`, `step`, …).  State tensors are sharded identically to their parameters.

**Important**: all ranks participate in every all-to-all (collective requirement), but
only **target owner** ranks update their state dict.  Non-owner ranks keep their
(now stale) state unchanged rather than having it overwritten with zeros.

### Phases 6–8 — Group management and live-reference updates

6. Update plugin config; install new pg_mesh; call `_init_communication_groups()` to
   rebuild stage_manager, process groups, shard_config, and scheduler.
7. Destroy stale groups: `old_pg_mesh.destroy_stale_groups(set(new_pg_mesh._ranks_to_group))`.
   Groups that are reused by the new mesh are kept alive.
8. Update live references on the model wrapper (`dp_group`, `tp_group`, `sp_group`,
   `stage_manager`) and on the optimizer (`tp_pg`, `pp_pg`).

---

## Communication Summary

| Old design | New design |
|------------|------------|
| `all_gather` across TP group (unshards) | *(eliminated)* |
| `all_to_all` with **full-tensor** sized buffers | `all_to_all` with **piece-sized** buffers |
| `broadcast`/`scatter` across new TP group (reshards) | *(eliminated)* |
| Only TP rank 0 sends/receives per DP replica | All DP replicas are eligible senders |

The new design uses **one** `all_to_all` per parameter per reconfiguration, with
buffers of size `full_size / lcm(src_tp, tgt_tp)` instead of `full_size`.

---

## Files

| File | Role |
|------|------|
| `data_structures.py` | `LayerOwnership`, `TransferPiece`, `AllToAllPlan` |
| `ownership_analyzer.py` | `TensorOwnershipAnalyzer`, `build_target_ownership` |
| `executor.py` | `ReconfigurationExecutor` — piece computation, all-to-all, assembly |
| `utils.py` | `get_param_by_name`, `set_param_by_name`, `get_all_param_names` |
| `tp_handler.py` | `TPReconfigurationHandler` — legacy gather/scatter helpers (no longer called by `reconfigure()`) |
