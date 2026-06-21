## Task ID: T002-PARALLELISM
## Type: IMPLEMENTATION (build the parallelism interface; tests must be green)
## Goal: implement the Option B+C parallelism interface chosen in specs/drafts/parallelism_interface_options.md

Build the two-layer parallelism interface for the new `cornstarch/` package:
- **Layer 1 (foundation, Option B):** composable functional primitives —
  process-group mesh + `apply_*` model-side parallelism + training schedule +
  gradient synchronizer + CP splitters.
- **Layer 2 (surface, Option C):** declarative per-modality `ParallelConfig` +
  `ParallelizationPlan.distribute()` returning a context that folds DP/CP into
  the dataloader, builds the schedule, and exposes gradient sync.

The foundation already exists on the `refactor_distributed` branch — **harvest
and port it**, do not rewrite from scratch. Apply the human decisions below.

## Human decisions (from specs/drafts/parallelism_interface_options.md → HUMAN DECISION)
- Chosen design: **Option B + Option C** (foundation primitives + declarative plan).
- **EP is a mesh axis:** promote expert parallelism into `ModalProcessGroupMesh`
  (currently a separate whole-world group). EP must compose with DP/CP/TP/PP.
- **TP backend:** keep **DTensor for TP only**. Do not extend DTensor to PP/FSDP2.
- **Silent TP no-op:** replace "unregistered model family → no-op" with an
  **explicit error**.
- **Numerical-equivalence tests:** add single-GPU-vs-parallel loss/grad parity
  tests. They need not be bit-identical — assert **closeness with atol/rtol = 1e-3
  on bf16**.
- **Checkpointing: OUT OF SCOPE.** Do not implement sharded save/load now. Record
  it in `tasks/backlog.md` to resume later.
- **ZeRO / optimizer-state sharding across DP: OUT OF SCOPE.** Record in
  `tasks/backlog.md`.

## What you must read
- specs/drafts/parallelism_interface_options.md   # the chosen design + HUMAN DECISION section
- specs/context/parallelism_background.md          # the five framework constraints
- The `refactor_distributed` branch (read via `git show refactor_distributed:<path>`, do NOT checkout):
  - cornstarch/distributed/parallel_config.py, process_group_mesh.py
  - cornstarch/distributed/tensor_parallel/{__init__.py, plans.py}
  - cornstarch/distributed/pipeline_parallel/{__init__.py, schedule.py, p2p.py, forward_spec_wrapper.py}
  - cornstarch/distributed/expert_parallel/{__init__.py, routing.py}
  - cornstarch/distributed/context_parallel/*  and  cornstarch/distributed/data_parallel.py
  - cornstarch/models/multimodal/parallelization.py   # the unfinished Option C facade
  - cornstarch/models/model_base.py                   # _section_names(), DTensor-aware/dtype-aware materialize
  - examples/distributed/pretrain_{llm,vlm,valm,llm_moe}.py
  - tests/distributed/*
- Current cornstarch integration seams: cornstarch/models/model_base.py,
  language_model.py, encoder_base.py, forward_specs.py, lazy_init.py,
  cornstarch/models/multimodal/{execution.py, modeling.py}

## What you must produce
1. **`cornstarch/distributed/` package** on the current branch, ported from
   `refactor_distributed` (do NOT import from `cornstarch_old`):
   - `ModalProcessGroupMesh` extended so **EP is a real mesh axis** that composes
     with DP/CP/TP/PP (resolve the per-modality + DP-offset + EP rank math).
   - `apply_tensor_parallel` (DTensor-based, **raises an explicit error** for
     unregistered model families instead of silently no-op'ing),
     `apply_context_parallel`, `apply_pipeline_parallel`, `apply_expert_parallel`.
   - `TrainingSchedule` hierarchy (`NonPipelineParallelSchedule`,
     `OneForwardOneBackwardSchedule`) + P2P.
   - `GradientSynchronizer` (bucketed DP all-reduce; skips `_is_expert_parallel`).
   - Context-parallel splitters (e.g. `UniformContextParallelSplitter`) and the
     model-side attention shim.
2. **Option C surface:** a clean `ParallelConfig` (no `cornstarch_old.PipelineTemplate`
   dependency) + `ParallelizationPlan` with `.parallelize(module, config)` and
   `.distribute(device, dtype)` returning a context that exposes
   `prepare_dataloader` (DP sampler + CP split), `create_schedule`, and
   `sync_gradients`. This is the surface the examples use.
3. **Model-base integration points** in `cornstarch/`: `_section_names()`,
   DTensor-aware `_materialize_empty`, `materialize(dtype=...)` — ported as needed
   so the `apply_*` functions work generically over the three-section layout.
4. **Examples** under `examples/distributed/` that use the **Option C surface**
   (deduplicated — no copy-pasted `_apply_parallelism`/`_build_plan`/rank-math
   across scripts).
5. **Tests** under `tests/distributed/`:
   - per-parallelism tests (DP, CP, TP, PP, EP),
   - the EP-as-axis composition (EP together with TP/PP/CP),
   - **numerical-equivalence tests**: single-GPU baseline vs parallel,
     `torch.testing.assert_close(..., atol=1e-3, rtol=1e-3)` on bf16,
   - an explicit-error test for `apply_tensor_parallel` on an unregistered family.
6. Append the deferred work (checkpoint gather/scatter + sharded lazy load; ZeRO
   optimizer-state sharding) to **`tasks/backlog.md`**.

## What you must NOT do
- Do NOT implement Option A (Plugin/Booster god-object).
- Do NOT implement Option D (parallelism config threaded through `from_hf_config`).
- Do NOT implement checkpointing (sharded save/load) or ZeRO — defer to backlog.
- Do NOT import from `cornstarch_old` (including `PipelineTemplate`). Duplicate if needed.
- Do NOT extend DTensor beyond TP sharding (no DTensor-based PP/FSDP2).
- Do NOT add global state (no module-level `dist.init_process_group()`).

## On Ambiguity
Per CLAUDE.md: write your interpretation to this file under a BLOCK section and
stop. Do not guess on interface decisions.

## Done Condition
- `cornstarch/distributed/` provides the Option B primitives (with EP as a mesh
  axis, DTensor-only TP, explicit error on unregistered TP family) AND the Option
  C `ParallelConfig` / `ParallelizationPlan` surface, with no `cornstarch_old` imports.
- `examples/distributed/` uses the Option C surface without per-script duplication.
- `pytest tests` is green, including new `tests/distributed/` numerical-equivalence
  tests (atol/rtol 1e-3, bf16) and the unregistered-family error test.
- Deferred checkpointing and ZeRO work is recorded in `tasks/backlog.md`.

---
## Completion
- Status: DONE
- PR: https://github.com/cornstarch-org/Cornstarch/pull/69 (targets refactor_parallelism, not merged)
- `pytest tests`: 142 passed, 1 skipped (CP numerical equivalence needs >=2 GPUs).
- Deferred work (sharded checkpointing, ZeRO, linear-attn-as-PP-boundary) recorded in tasks/backlog.md (B1/B2/B3).
