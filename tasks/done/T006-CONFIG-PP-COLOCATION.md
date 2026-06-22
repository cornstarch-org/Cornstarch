## Task ID: T006-CONFIG-PP-COLOCATION
## Type: IMPLEMENTATION (`pytest tests` must stay green)
## Goal: make the user's pipeline-parallelism intent **explicit** through
`ParallelConfig.pipeline_parallel_size`, and have `ParallelizationPlan.materialize`
assign ranks accordingly: **co-locate** all modules on shared ranks when PP is not
used, **disaggregate** them onto disjoint rank ranges when it is. Cornstarch must
**never infer** how many stages a module needs.

## Motivation
- Today every modality always gets a *disjoint* rank range
  (`ranks_per_replica = sum(cfg.ranks_per_replica)`), so even a VLM with no
  pipeline parallelism is split across meshes and a bare
  `output_future.execute()` on an encoder-only rank hits the language model on
  `meta`. The downstream goal (T008: "no schedule when PP is not used") needs the
  no-PP case to run encoder→merge→LLM **locally on every rank**, i.e. modules
  co-located.
- Whether to co-locate or disaggregate is the user's call and must be explicit;
  Cornstarch must not guess stage counts.

## Design decision (CONFIRMED with the user)
`ParallelConfig.pipeline_parallel_size`:
- `None` (new default) → this module is **not** a pipeline stage → **co-locate**.
- positive int `k` → this module occupies `k` pipeline stages → **disaggregate**.

Across all registered modules (checked in `materialize`):
- **all `None`** → no PP → every module **co-located** on the same replica rank
  range (each rank runs every modality; `num_pp_stages == 1`).
- **all positive** → PP used → modules **disaggregated** onto disjoint rank
  ranges (the current `sum(ranks_per_replica)` layout).
- **any mix** → raise a clear error.

`ParallelContext` exposes **`uses_pipeline_parallel: bool`** (True iff
disaggregated/PP) so T008's call sites can branch (schedule vs direct execute).

## What you must read
- `cornstarch/distributed/parallel_config.py`
  (`ParallelConfig` fields, `__post_init__` validation, `ranks_per_replica`).
- `cornstarch/distributed/parallelization.py`
  (`ParallelizationPlan.materialize` rank math + `_build_dp_handles`;
  `ParallelContext` accessors / `_meshes` / `_layouts`; `MeshLayout`).
- `cornstarch/distributed/process_group_mesh.py`
  (`ModalProcessGroupMesh` constructor: `(dp, pp, cp, tp, ep)` reshape).
- `tests/distributed/test_parallelism_integration.py` (passes
  `pipeline_parallel_size=pp ∈ {1,2}`),
  `tests/distributed/test_parallelization_plan.py` (`ParallelConfig()` /
  `ParallelConfig(tensor_parallel_size=...)` with default pp),
  `tests/distributed/test_multimodal_distributed.py` (T005 VLM, currently
  disjoint), `tests/distributed/test_numerical_equivalence.py`,
  `examples/distributed/pretrain_llm.py` / `pretrain_vlm.py`.

## Commits
### Commit 1 — `ParallelConfig.pipeline_parallel_size` becomes `int | None = None`
- `parallel_config.py`: type → `int | None`, default `None`. `__post_init__`:
  skip the `>= 1` check when `None`; keep `>= 1` for the other sizes and for a
  non-`None` pp. `ranks_per_replica`: treat `None` as `1`.
- Docstring: document `None` = co-locate (no PP) vs positive = disaggregate (PP).

### Commit 2 — PP-driven rank assignment + `uses_pipeline_parallel`
- In `materialize`, classify the registered configs:
  - all `pipeline_parallel_size is None` → **co-located**.
  - all positive ints → **disaggregated**.
  - mixed → `raise ValueError(...)` naming the offending modules.
- **Disaggregated** (unchanged behavior): per-modality disjoint ranges via
  `sum(ranks_per_replica)`, `num_pp_stages = cfg.pipeline_parallel_size`.
- **Co-located:** all modules share one replica rank range. Require every module's
  `ranks_per_replica` (== `tp*cp*ep`, since pp is `None`→1) to be **equal**
  (`R`); else raise (mixed-degree co-location / replication is out of scope —
  state it). `dp = world_size / R`. Each module's `MeshLayout`/mesh is built over
  the **same** global ranks with `(dp, num_pp_stages=1, cp, tp, ep)`, so every
  rank belongs to every module's mesh and materializes every module.
- `ParallelContext`: store the mode and expose
  `uses_pipeline_parallel: bool` (True for disaggregated). `_build_dp_handles`
  stays correct in both modes (co-located: all meshes share the same dp group).
- Keep `MeshLayout` (added in T005) populated for every module in both modes.

### Commit 3 — converge call sites to the explicit pp interface
- `examples/distributed/pretrain_llm.py` / `pretrain_vlm.py`: pass
  `pipeline_parallel_size=None` when the user wants no PP (default), a positive int
  to pipeline. (Do not change the schedule wiring here — that is T008; this commit
  only makes the configs explicit and keeps the examples running.)
- `tests/distributed/test_parallelism_integration.py`: the non-PP combos should
  use `pipeline_parallel_size=None` (co-located, no PP); the PP combos keep a
  positive value. `test_parallelization_plan.py`: `ParallelConfig()` now means
  co-locate — adjust expectations if any.
- `test_multimodal_distributed.py` (T005): its VLM currently relies on disjoint
  ranks; set the configs to positive `pipeline_parallel_size` so it stays
  disaggregated (the cross-mesh `CompiledSchedule` path is unchanged until T008).

## Tests
- `ParallelConfig`: `pipeline_parallel_size` defaults to `None`; `None` →
  `ranks_per_replica` treats it as 1; negative/zero still rejected.
- `materialize` mode classification: all-`None` co-located, all-positive
  disaggregated, mixed raises; `ctx.uses_pipeline_parallel` reflects it.
- Co-located run (gloo CPU, `world_size<=2`): a 1-module LM and a 2-module VLM
  (equal `ranks_per_replica`) materialize every module on every rank and one
  `execute()`+`backward()` produces finite loss + grads on every rank.
- Co-located with unequal `ranks_per_replica` raises a clear error.
- Existing disaggregated tests (T005 VLM, PP integration, numerical equivalence)
  stay green.
- `pytest tests` green.

## What you must NOT do
- Do NOT infer stage counts; only the user's `pipeline_parallel_size` decides.
- Do NOT change CP/TP/PP/EP/DP numerical behavior.
- Do NOT touch the schedule classes (that is T008); this task is rank assignment +
  config only. (`create_schedule` may still be called as today by existing
  tests/examples; leave it until T008.)
- Do NOT import from `cornstarch_old`.

## Done Condition
- `pipeline_parallel_size` is `int | None`, default `None`, with the co-locate /
  disaggregate / mixed-error semantics; `ctx.uses_pipeline_parallel` exposed.
- Co-located materialization puts every module on every replica rank; the no-PP
  VLM can run encoder→merge→LLM locally.
- `pytest tests` green; branch pushed; PR opened against `feat/T002-parallelism`;
  this file moved to `tasks/done/T006-CONFIG-PP-COLOCATION.md` with the PR URL.

## HUMAN COMMENTS
- PR targets `feat/T002-parallelism`; auto-merge into it after green so T007/T008
  build cleanly (the user authorized auto-merging the prerequisites).
