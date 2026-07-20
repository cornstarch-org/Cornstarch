## Task ID: T005-PER-STEP-EXECUTION-PLAN
## Type: DESIGN + IMPLEMENTATION (`pytest tests` must stay green)
## Goal: make `CornstarchExecutionPlan` usage identical for non-distributed and
distributed training. The plan is rebuilt **per training step** (recovering the
per-batch flexibility the non-distributed example already has) and is purely a
*logical* data-flow DAG that is agnostic to distributed training. All knowledge
of *which ranks are active* and *how data crosses ranks* is compiled into a
`TrainingSchedule` that is generated **per batch** from that plan + the
`ParallelContext` meshes. Runtime just runs the compiled schedule.

## Motivation
- Non-distributed `examples/pretrain_vlm.py::_training_step` builds a fresh
  `CornstarchExecutionPlan` **every step** and calls
  `output_future.execute(inputs=batch)`. This gives per-batch flexibility: a step
  with no image simply omits the `run_modality_encoder("vision")` node.
- Distributed `examples/distributed/pretrain_*.py` build the plan **once** before
  the loop, bake it into a `TrainingSchedule` via `ctx.create_schedule(...)`, and
  reuse that schedule for every step. The plan is never regenerated, so the
  per-batch flexibility is lost and the execution DAG is frozen.
- Worse, the current `NonPipelineParallelSchedule.step` runs the **whole DAG on
  every rank** (`output_future.execute`). When modalities live on **disjoint
  ranks** (Option C assigns each modality its own rank range), the vision encoder
  is materialized only on the vision ranks and is still `meta` on the LLM ranks,
  so running the full DAG there raises `Tensor on device cpu is not on the
  expected device meta!`. This is the limitation recorded at the end of
  `tasks/done/T004-DISTRIBUTED-UX.md` (commit "distributed examples mirror the
  non-distributed style"); the VLM example parallelizes/materializes correctly
  and reaches `schedule.step`, then fails here.

This task fixes both: per-step plan rebuild **and** a schedule that compiles
per-batch rank activation + cross-rank data movement from the DAG.

## Design decision (records the user's choice)
Two options were considered for the schedule:
1. **Build one static schedule, activate a sub-DAG per step.** The schedule holds
   the full model map; each step it analyzes which sub-DAG/ranks to activate.
2. **Build a fresh schedule per batch (CHOSEN).** Each step builds a
   `CornstarchExecutionPlan` for that batch; a schedule is compiled from that DAG
   (which ranks run which node, and the P2P/broadcast edges that move a node's
   output to the ranks that consume it). Runtime just runs the compiled schedule.

Choice **2** is preferred because it makes `CornstarchExecutionPlan` usage
identical whether or not the model is distributed: the plan is the logical data
flow only, and *how* data is transferred under parallelism is governed entirely
by the schedule. Per-step schedule construction must therefore be cheap (DAG
analysis + a rank-local program, no collectives at build time).

**Invariant to preserve:** `cornstarch/models/multimodal/execution.py` must not
import anything from `cornstarch/distributed/`. `CornstarchExecutionPlan` /
`ExecutionNode` / `ExecutionFuture` stay distributed-agnostic.

## What you must read
- `cornstarch/models/multimodal/execution.py`
  (`CornstarchExecutionPlan`, `ExecutionNode` {`name`, `kind`, `params`,
  `dependencies`}, `ExecutionFuture.execute`, `_topological_nodes`,
  `_execute_nodes`, `_execute_node`, `_execute_until`) — the logical DAG.
- `cornstarch/distributed/parallelization.py`
  (`ParallelContext` — holds `self._meshes: dict[id(module) -> ModalProcessGroupMesh]`,
  `self._modules`, `create_schedule`, `sync_gradients`; `ParallelizationPlan.materialize`).
- `cornstarch/distributed/pipeline_parallel/schedule.py`
  (`TrainingSchedule`, `NonPipelineParallelSchedule.step` = execute+criterion+
  backward, `OneForwardOneBackwardSchedule` = 1F1B with `_StageModel` splitting
  the DAG into pre-pipeline nodes + the LM node, `PipelineP2PCommunication`).
- `cornstarch/distributed/pipeline_parallel/p2p.py` (P2P send/recv primitives —
  the cross-rank transport to reuse for cross-mesh edges).
- `cornstarch/distributed/process_group_mesh.py` (`ModalProcessGroupMesh`: per
  modality, the dp/cp/tp/pp/ep axes, `is_first_stage`/`is_last_stage`,
  `num_stages`, the rank set).
- `examples/pretrain_vlm.py` (`_training_step` builds the plan per step) and
  `examples/distributed/pretrain_vlm.py` / `pretrain_llm.py` (build the plan once
  today — the call sites to converge).
- `tests/distributed/test_parallelism_integration.py` (LM + EP composition over
  the surface) and `tests/distributed/test_numerical_equivalence.py` (TP/PP/EP/DP
  parity harness) — the green bar this must not regress.

## Proposed approach (refine during implementation; write a BLOCK if an
## interface decision is unclear, per CLAUDE.md)

### Commit 1 — per-step plan rebuild in the distributed examples (no engine change yet)
Move plan construction into `_training_step` for the LM-only case (single mesh,
no cross-rank modality edge), so the example builds a fresh plan per step exactly
like the non-distributed one, and the schedule is created per step from it.
- `examples/distributed/pretrain_llm.py`: build `CornstarchExecutionPlan` +
  `output_future` inside `_training_step` (or just before it), call
  `ctx.create_schedule(plan, output_future, ...)` per step, then `schedule.step`.
- Confirm `ctx.create_schedule` is cheap enough per step for the non-PP and PP
  LM paths (it already only inspects the DAG). If PP per-step schedule rebuild is
  too costly or stateful, note it and keep PP schedule cached (LM topology is
  batch-invariant) while still rebuilding the *plan*.
- Smoke-check the 2-rank gloo LM run still trains.

### Commit 2 — schedule compiles per-batch rank activation + cross-mesh edges
Teach the schedule layer to run a DAG whose nodes live on **different meshes**
(disjoint rank ranges), which is what fixes the multimodal cross-rank failure.
- Give `ParallelContext` a map from execution node -> owning mesh: a node's
  module is in `node.params["module"]`; look up `self._meshes[id(module)]`. A
  rank executes a node iff it is in that mesh's rank set; otherwise the node is a
  no-op on that rank.
- Compile cross-mesh edges from the DAG: when node A (mesh M_A) produces a future
  consumed by node B (mesh M_B) and `M_A != M_B`, insert a transfer — the
  designated producer rank(s) of M_A `send` the tensor and the consumer rank(s)
  of M_B `recv` it (reuse `PipelineP2PCommunication` / `p2p.py`). Define the
  rank-pairing rule (e.g. M_A last-PP-stage rank r -> M_B first-PP-stage rank r,
  with DP/TP broadcast as needed) and record it; this is the core interface
  decision — if it is ambiguous, STOP and write a BLOCK.
- A rank with no active node for a batch (e.g. vision ranks on a text-only step)
  does nothing. Backward mirrors the forward transfers (send grad back across the
  cross-mesh edge).
- Keep the existing single-mesh fast paths (`NonPipelineParallelSchedule` for one
  non-PP module, `OneForwardOneBackwardSchedule` for one PP module) as the
  degenerate cases of the compiled schedule, or subsume them — decide and state.

### Commit 3 — converge the distributed VLM example onto the per-step plan
- `examples/distributed/pretrain_vlm.py`: build the multimodal
  `CornstarchExecutionPlan` per step inside `_training_step`, create the
  per-batch schedule, run it. The plan-construction lines must be byte-identical
  to `examples/pretrain_vlm.py` (only the execute-vs-schedule.step tail differs).
- Demonstrate flexibility: a step whose batch lacks `pixel_values` builds a plan
  with no vision node; vision ranks idle, LLM ranks train. Add this as the
  motivating comment/example.
- Smoke-check the 4-rank (or 2-rank gloo) VLM run trains end-to-end — this is the
  acceptance signal that the cross-rank limitation is resolved.

### Tests
- A `tests/distributed/` test that builds a tiny VLM (vision encoder + LM on
  disjoint ranks), parallelizes both via the Option C surface, builds the plan
  per step, and asserts one forward+backward step produces a finite loss and
  gradients on both meshes (mirror `_run_combo` in
  `test_parallelism_integration.py`). This is the regression guard the VLM path
  currently lacks.
- A flexibility test: alternating batches with/without the vision modality both
  run without error and only touch the expected parameters' grads.
- Keep TP/PP/EP/DP numerical-equivalence green (no math change).

## What you must NOT do
- Do NOT make `cornstarch/models/multimodal/execution.py` import from
  `cornstarch/distributed/` — the plan stays distributed-agnostic.
- Do NOT change CP/TP/PP/EP/DP numerical behavior; this is a scheduling/ergonomics
  change. Cross-mesh transfer must be mathematically transparent.
- Do NOT make the non-distributed `examples/pretrain_vlm.py` depend on the
  distributed package (convergence is distributed -> non-distributed).
- Do NOT introduce a separate checkpoint format.

## On Ambiguity
Per CLAUDE.md: the cross-mesh rank-pairing/broadcast rule (Commit 2) is the main
interface decision. If it is unclear how a producer mesh's output maps onto a
consumer mesh's ranks under combined DP/TP/PP, write the interpretation under a
BLOCK section here and stop rather than guessing.

## Done Condition
- Distributed `_training_step` builds `CornstarchExecutionPlan` per step; the
  plan-construction code matches the non-distributed example.
- A schedule is compiled per batch and governs all cross-rank data movement; the
  execution plan contains zero distributed concepts.
- The distributed VLM example trains end-to-end on disjoint modality ranks (the
  T004 cross-rank limitation is gone); a batch without a modality idles that
  modality's ranks.
- New distributed VLM step + flexibility tests are green; TP/PP/EP/DP equivalence
  unchanged; `pytest tests` green.
- Branch pushed; PR opened against the appropriate base; this file moved to
  `tasks/done/T005-PER-STEP-EXECUTION-PLAN.md` with the PR URL.

## IMPLEMENTATION NOTES (recorded interface decision — Commit 2)

The cross-mesh rank-pairing rule (the flagged ambiguity) was resolved as the
cross-mesh analogue of a pipeline-stage boundary, encoded in `MeshLayout` +
`CompiledSchedule` (`cornstarch/distributed/pipeline_parallel/schedule.py`):

- A node executes on a rank iff that rank is in the node's owning mesh
  (`run_modality_encoder` -> the encoder's mesh; `merge_*` and
  `run_language_model` -> the language model's mesh). A rank with no active node
  for a batch idles — this is the per-batch flexibility (text-only step).
- For a cross-mesh edge `A (mesh M_A) -> B (mesh M_B)`, transfers pair
  **data-parallel replica `d` of `M_A` with replica `d` of `M_B`** (dp size is
  identical across modalities by construction, so each replica's batch shard
  stays local):
  - **Forward:** the producer *representative* of `M_A` for replica `d` — its
    last-PP-stage `(cp=tp=ep=0)` rank — sends the feature tensor to *every*
    first-PP-stage rank of `M_B` in replica `d`. The fan-out broadcast is needed
    because the consuming layer needs the value replicated across the consumer's
    TP/EP group; TP's column-parallel input is replicated in forward.
  - **Backward:** mirrored. The consumer representative of `M_B` sends the
    gradient back to *every* last-PP-stage rank of `M_A` in replica `d`; each
    runs its local `backward` into the producing subgraph. Taking the gradient
    from a single consumer representative is correct because TP all-reduces the
    input gradient in backward (all consumer TP ranks hold the same gradient).
- Transport reuses the PP P2P primitives, generalized to explicit global ranks
  (`exchange_objects` in `p2p.py`). A cross-mesh edge is a graph break by design
  (like a PP boundary): the received tensor is a fresh leaf and its gradient is
  shipped back explicitly. The transfer is mathematically transparent — no
  CP/TP/PP/EP/DP math changes.

**Scope:** every mesh in a *multi-mesh* plan must be non-pipelined
(`num_pp_stages == 1`); `CompiledSchedule` raises `NotImplementedError`
otherwise. Single-mesh pipelining is unchanged (`OneForwardOneBackwardSchedule`).
Combining intra-mesh 1F1B with cross-mesh microbatch coordination is left out
(the VLM example default is `--llm-pp 1`). CP across a cross-mesh edge (sequence
sharding of merged modality features) is likewise out of scope and untested.

The `execution.py` invariant is preserved: it imports nothing from
`cornstarch/distributed/`; all cross-rank logic lives in the schedule.

## HUMAN COMMENTS
- When done, PR should target `feat/T002-parallelism` branch.

## PR
https://github.com/cornstarch-org/Cornstarch/pull/72 (base: feat/T002-parallelism)