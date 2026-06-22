## Task ID: T008-SCHEDULE-CONSOLIDATION
## Type: DESIGN + IMPLEMENTATION (`pytest tests` must stay green)
## Depends on: T006 (PP-driven rank co-location) and T007 (microbatch via
## `collate_fn`), both merged into `feat/T002-parallelism` first; branch T008 from
## the post-T007 `feat/T002-parallelism`.
## Goal: collapse the schedule hierarchy to **two** classes under
`cornstarch/distributed/pipeline_parallel/`: an extensible **base pipeline
schedule** and a **1F1B** subclass. A schedule exists **only when pipeline
parallelism is in play**; when PP is not used there are no stages and no
schedule at all — training runs `output_future.execute()` + `loss.backward()`,
exactly like the non-distributed `examples/pretrain_vlm.py`. **When PP is used**,
each modality encoder becomes a **leading pipeline stage** feeding the
language-model stages, and the schedule **microbatches all inputs and overlaps
execution following 1F1B**. The stage set for a step is assembled **per step
from that step's plan** (image steps run encoder+LLM stages; text-only steps run
LLM-only stages with the encoder ranks idle); the microbatch count is whatever the user's `collate_fn`
returns (Prerequisite 2), while warmup/steady/cooldown follow the stages present.

## Prerequisites (already implemented + merged into `feat/T002-parallelism`)
- **T006** — `ParallelConfig.pipeline_parallel_size: int | None = None`; all-`None`
  ⇒ co-located (no PP), all-positive ⇒ disaggregated (PP), mix ⇒ error;
  `ctx.uses_pipeline_parallel` exposed.
- **T007** — `collate_fn` returns `list[dict]` microbatches; `prepare_dataloader`
  applies DP/CP per microbatch and yields the list; the non-PP training path does
  gradient accumulation over the list (no schedule).
This task consumes both: `create_schedule` is only called when
`ctx.uses_pipeline_parallel`, and `schedule.step` takes the microbatch list.

## Motivation
- `TrainingSchedule` lives under `pipeline_parallel`, but today a schedule is
  produced even when no pipeline parallelism is used:
  `NonPipelineParallelSchedule` just wraps `output_future.execute()` +
  `criterion` + `backward`. That is not a "schedule"; it is the plain
  non-distributed step. It should not exist — the non-PP path should call
  `execute()`/`backward()` directly.
- T005 added `CompiledSchedule` for the multi-mesh (encoder on disjoint ranks)
  case, but it runs a **single full-batch** forward/backward with no
  microbatching and explicitly bails on PP (`NotImplementedError` for
  `num_pp_stages > 1`). So a VLM cannot pipeline, and there are now **three**
  schedule classes with overlapping responsibilities
  (`NonPipelineParallelSchedule`, `OneForwardOneBackwardSchedule`,
  `CompiledSchedule`).
- The unifying insight: a modality encoder on its own ranks feeding the language
  model **is** a pipeline-stage boundary. Treating the encoder as the first
  pipeline stage makes the cross-mesh transfer just another stage boundary, lets
  the VLM pipeline its microbatches, and removes the need for a separate
  multi-mesh schedule.

## Design decision (CONFIRMED with the user — record, do not re-litigate)
**A schedule exists only when pipeline parallelism is used. When PP is used, the
encoder is a genuine leading pipeline stage.**

"Pipeline parallelism is used" = the encoder is split from the language model
into separate pipeline stages, and/or the language model itself is split across
stages (`num_pp_stages > 1`).

- **When PP is NOT used:** there are **no pipeline stages at all**. All
  modalities are **co-located on every rank** — a single rank runs encoder →
  merge → language model directly (DP/TP optional). The step is a plain
  `output_future.execute()` + `loss.backward()` + `ctx.sync_gradients()`,
  exactly like the non-distributed `examples/pretrain_vlm.py`. **No schedule
  object is created and no activation is transferred across ranks.**
  *(This requires modalities to be co-located rather than on disjoint ranks —
  see "OPEN: rank co-location" below; it is the one real ripple of this rule.)*

- **When PP is used:** the modalities form a pipeline whose **depth = (encoder
  stages) + (language-model stages)**. The encoder is the leading stage(s) (one
  stage per encoder; intra-encoder PP is out of scope), the language model the
  following stages, on disjoint ranks; the encoder→LLM boundary is a
  pipeline-stage boundary. 1F1B microbatches and overlaps — e.g. with one
  encoder stage and `llm_pp == 1` there are 2 stages, so the encoder of
  microbatch `k+1` overlaps the LLM of microbatch `k`.

**Text-only batch flexibility (T005) — revised per the user (supersedes the
earlier "w/s/c invariant to encoder inputs" wording):** the pipeline structure
for a step is **assembled per step from that step's plan** (which modalities are
present) using the per-modality **stage shapes from config**. A text-only step's
plan has no `run_modality_encoder` node, so **the encoder stage is simply absent
for that step**: the encoder ranks do nothing, and the first language-model stage
begins its 1F1B computation immediately without waiting for or receiving any
encoder activation. **No empty/placeholder activation is sent.**

This does NOT sacrifice consistency: each `step()` runs one self-contained 1F1B
forward+backward, and every rank derives the same stage set from the same
per-step plan (all ranks see the same batch), so there is no rank disagreement
or deadlock. The microbatch count is `len(microbatches)` (the user-provided list
from `collate_fn`, per Prerequisite 2) and is the same for every rank;
warmup/steady/cooldown reflect the stages actually present that step (encoder+LLM
on image steps, LLM-only on text steps). That per-step variation is fine because
steps share no pipeline state. Schedule construction must stay cheap (DAG/stage
analysis from the per-step plan + config, no collectives).

  *(Co-location is handled by **Prerequisite 1** above: when PP is not used all
  modules share one rank range, so `output_future.execute()` runs the full
  encoder→merge→LLM locally with no transfer and no schedule.)*

## What you must read
- `cornstarch/distributed/pipeline_parallel/schedule.py`
  (`TrainingSchedule`, `NonPipelineParallelSchedule`,
  `OneForwardOneBackwardSchedule` — `_StageModel`, `_run_1f1b`,
  `_forward_step`/`_backward_step`, warmup/steady/cooldown math —, `MeshLayout`,
  `CompiledSchedule` — `_forward_transfer`/`_backward_transfer` seam pairing).
- `cornstarch/distributed/pipeline_parallel/p2p.py`
  (`PipelineP2PCommunication` for intra-mesh stage hops; `exchange_objects` for
  explicit-global-rank seam transfers — reuse both).
- `cornstarch/distributed/process_group_mesh.py`
  (`ModalProcessGroupMesh`: `stage`, `num_stages`, `is_first_stage`,
  `is_last_stage`, `get_prev_ranks`/`get_next_ranks`, `distribute_layers`).
- `cornstarch/distributed/parallelization.py`
  (`ParallelContext.create_schedule` selection logic, `_meshes`, `_layouts`,
  `materialize` building `MeshLayout` per module).
- `cornstarch/models/multimodal/execution.py`
  (`CornstarchExecutionPlan` nodes/kinds, `_execute_node`, `_topological_nodes`,
  `_first_output_tensor`, `merge_modality_encoder_outputs` — note it masks labels
  at modality-token positions to -100).
- `examples/distributed/pretrain_llm.py` and `pretrain_vlm.py`
  (`_training_step` builds the plan per step and calls `ctx.create_schedule` +
  `schedule.step`; these are the call sites to converge).
- `examples/pretrain_vlm.py` (the non-distributed step to mirror: `execute()` +
  `backward()` with no schedule).
- `tests/distributed/test_parallelism_integration.py` (calls
  `create_schedule`/`step` for non-PP and PP LM combos — must keep passing),
  `tests/distributed/test_numerical_equivalence.py` (TP/PP/EP/DP parity — no math
  change), `tests/distributed/test_multimodal_distributed.py` (T005 cross-mesh +
  flexibility tests — must keep passing, adapted to the new surface).

## Proposed approach (refine during implementation; write a BLOCK per CLAUDE.md
## if an interface decision below is unclear)

### Commit 1 — base pipeline schedule + 1F1B subclass (no behavior change yet)
Refactor `OneForwardOneBackwardSchedule` into:
- `BasePipelineSchedule(TrainingSchedule)` — owns everything schedule-shape-
  independent: the stage model, **consuming the user-provided microbatch list**
  (`step(microbatches: list[dict], ...)`, `num_microbatches = len(microbatches)`;
  drop the internal `_get_micro_batch`/`microbatch_size` slicer per Prereq 2),
  per-stage forward/backward steps, the P2P send/recv wrappers, loss
  accumulation, and the abstract iteration order. Document it as the extension
  point for other pipeline schedules (e.g. a future GPipe — **do not implement
  GPipe**).
- `OneForwardOneBackwardSchedule(BasePipelineSchedule)` — implements only the
  1F1B warmup/steady/cooldown ordering.
- `ctx.create_schedule` drops the `num_microbatches`/`microbatch_size` params
  (the count comes from the microbatch list passed to `step`).
- Keep the single-mesh LM PP path behaviorally identical (same loss/grads); the
  integration + numerical-equivalence PP tests must stay green (adapted to pass a
  microbatch list instead of `num_microbatches`).

### Commit 2 — encoder as a leading pipeline stage; remove `CompiledSchedule`
Generalize the pipeline so its stage list spans the encoder mesh(es) and the LLM
mesh, with the encoder(s) as the leading stage(s):
- Build the global stage list from the **config** (the `ParallelContext` meshes/
  layouts), ordered encoder(s) → LLM stages. Map plan nodes to stages:
  `run_modality_encoder` → the encoder stage; `merge_modality_encoder_outputs` →
  the first LLM stage (it needs the LLM embedding); `run_language_model` →
  pipelined across the LLM stages (as today).
- Two boundary kinds: **intra-mesh** stage hops use `PipelineP2PCommunication`
  (1:1, same cp/tp/ep position, as today); the **encoder→LLM seam** uses the
  `MeshLayout` producer-rep-broadcast / consumer-rep-collapse pairing (lift it
  out of `CompiledSchedule` — `_forward_transfer`/`_backward_transfer` — into a
  reusable seam transfer). The encoder stage's "send_forward" is the seam send;
  the LLM first stage's "recv_forward" is the seam recv; backward mirrors.
- **Per-microbatch flow** (microbatches come from the user's `collate_fn` list,
  Prereq 2 — the schedule does not slice): for each microbatch `m`, the encoder
  stage runs `encoder(m["pixel_values"])` → seam → the first LLM stage merges
  those features with `m["input_ids"]`/`m["labels"]`. The user guarantees the
  per-microbatch image-token / feature counts match (that is why splitting is
  theirs).
- **Text-only step** (per the revised flexibility decision): the per-step plan
  has no `run_modality_encoder` node, so **the encoder stage is absent** — the
  encoder ranks do nothing and the first LLM stage is the leading stage (it does
  not `recv_forward` from any encoder). No placeholder activation. All ranks agree
  because they share the per-step plan. (Define the seam rank-pairing under
  combined DP/TP/PP for the image case; if ambiguous, STOP and write a BLOCK.)
- **Masked-label correctness through PP for VLM (resolved):** `merge_*` masks
  labels at modality-token positions to -100, but under PP the loss is computed on
  the last stage from the microbatch's labels, not the merge output. Since every
  rank has the same per-microbatch inputs and builds the same plan, **the last
  stage recomputes the mask locally** — from `input_ids` + `modality_token_ids`
  (both available in the microbatch / plan) — and masks `labels` to -100 before
  the loss. No need to ship masked labels down the pipeline.
- Delete `CompiledSchedule`. The multi-mesh case is now the leading-stage
  pipeline.

### Commit 3 — no schedule when PP is not used; converge call sites
- **`create_schedule` always returns a real (pipeline) schedule and is only ever
  called when PP is used; it never returns `None`.** Whether PP is used is a
  property of the `ParallelContext` (decided in Prereq 1: all-`None`
  `pipeline_parallel_size` ⇒ co-located, no PP; all-positive ⇒ disaggregated, PP).
  Expose that as e.g. `ctx.uses_pipeline_parallel` (or `ctx.is_pipelined`) so the
  caller can branch.
- **The training step diverges by PP**, and `CornstarchExecutionPlan` stays
  parallelism-agnostic (the plan-build lines are identical):
  - **No PP** (co-located): no schedule object at all — iterate the `collate_fn`
    microbatch list running `output_future.execute(inputs=mb)` + scaled
    `loss.backward()` (gradient accumulation), then `ctx.sync_gradients()`.
  - **PP**: `schedule = ctx.create_schedule(plan, output_future)`;
    `schedule.step(microbatches, criterion, optimizer)`; then
    `ctx.sync_gradients()`.
  Structure the examples as two small functions (a non-PP step and a PP step)
  selected on `ctx.uses_pipeline_parallel`, with the shared `CornstarchExecutionPlan`
  build in common.
- Delete `NonPipelineParallelSchedule`.
- Update `examples/distributed/pretrain_llm.py` / `pretrain_vlm.py` and
  `tests/distributed/test_parallelism_integration.py` /
  `test_multimodal_distributed.py` to this surface.
- Update `cornstarch/distributed/__init__.py` exports (drop the removed classes,
  add `BasePipelineSchedule`; keep `MeshLayout`).

### Tests (same commit or immediately after the relevant impl)
- VLM 1F1B with `llm_pp=1` **and** `llm_pp>1`, with a multi-microbatch
  `collate_fn` (list length > 1): one forward+backward yields finite loss on the
  LLM last stage and gradients on both the encoder and LLM meshes (gloo CPU,
  `world_size<=4`). Mirror the existing
  `tests/distributed/test_multimodal_distributed.py` structure.
- Per-step flexibility: alternating image / text-only batches on the same VLM
  pipeline both run without deadlock. The image step runs encoder+LLM stages
  (vision does real work, gets grads); the text-only step runs LLM stages only
  with the encoder ranks idle (no vision grads, no seam transfer). The microbatch
  count (list length) is the same for both; the stage count / warmup-cooldown
  differ as expected (encoder stage present vs absent).
- Non-PP microbatch gradient accumulation (Prereq 2): a co-located run with a
  multi-microbatch `collate_fn` and no schedule accumulates gradients matching the
  single-batch reference.
- LM-only non-PP (dp/tp/dp+tp): no schedule is created; training still steps and
  produces grads.
- LM-only PP and TP/PP/EP/DP numerical equivalence: unchanged.
- Keep `pytest tests` green.

## What you must NOT do
- Do NOT make `cornstarch/models/multimodal/execution.py` import from
  `cornstarch/distributed/` — the plan stays distributed-agnostic.
- Do NOT change CP/TP/PP/EP/DP numerical behavior; seam + microbatching must be
  mathematically transparent.
- Do NOT implement GPipe — only leave the base class extensible for it.
- Do NOT make the non-distributed `examples/pretrain_vlm.py` depend on the
  distributed package.
- Do NOT introduce a separate checkpoint format.

## Resolved decisions / remaining ambiguity
Resolved (do not re-litigate):
- **Rank co-location** → Prerequisite 1.
- **Masked labels through PP for VLM** → last stage recomputes the mask locally
  (Commit 2).
- **`create_schedule` return** → always a real pipeline schedule, only used under
  PP; the non-PP path uses no schedule; the example step diverges by PP
  (Commit 3).

Remaining (write a BLOCK per CLAUDE.md if it turns out ambiguous during impl):
- The encoder→LLM **seam rank-pairing under combined DP/TP/PP** for the image
  case (Commit 2). Reuse the `MeshLayout` producer-rep-broadcast /
  consumer-rep-collapse rule lifted from `CompiledSchedule`; if a combination is
  unclear, STOP and write a BLOCK.

## Done Condition
- Exactly two schedule classes remain under `pipeline_parallel`:
  `BasePipelineSchedule` and `OneForwardOneBackwardSchedule`.
  `NonPipelineParallelSchedule` and `CompiledSchedule` are gone.
- No schedule is created for non-PP single-mesh training; those paths run
  `output_future.execute()` + `backward()` directly.
- When PP is used, a VLM trains with the encoder as a leading pipeline stage,
  microbatched and overlapped under 1F1B, at `llm_pp=1` and `llm_pp>1`; the stage
  set for a step is assembled from that step's plan (text-only steps run
  LLM-only stages, encoder ranks idle).
- When PP is not used, no schedule is created and the VLM step is a plain
  `execute()`/`backward()` (subject to the rank-co-location decision).
- New VLM 1F1B + per-step-flexibility tests are green; LM-only non-PP/PP and
  TP/PP/EP/DP equivalence unchanged; `pytest tests` green.
- Branch pushed; PR opened against the appropriate base; this file moved to
  `tasks/done/T006-SCHEDULE-CONSOLIDATION.md` with the PR URL.

## HUMAN COMMENTS
- (PR base branch? T005 merged into `feat/T002-parallelism`; confirm whether
  T006 should also target `feat/T002-parallelism`.)

## PR
https://github.com/cornstarch-org/Cornstarch/pull/75 (base: feat/T002-parallelism; left open for human review)
