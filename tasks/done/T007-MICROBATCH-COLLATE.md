## Task ID: T007-MICROBATCH-COLLATE
## Type: IMPLEMENTATION (`pytest tests` must stay green)
## Depends on: T006 (PP-driven rank co-location), merged into
## `feat/T002-parallelism` first; branch T007 from the post-T006 base.
## Goal: make microbatching a **user responsibility expressed through
`collate_fn`**, so Cornstarch never has to infer how to split modality tensors
(the old `ndim` heuristic could not handle e.g. a varying number of images per
sample). The user's `collate_fn` returns a **`list[dict]`** — the microbatches
for one optimizer step. Microbatching works **with or without** pipeline
parallelism; without PP it is plain gradient accumulation.

## Motivation
- A schedule (and a non-PP training loop) must split a batch into microbatches
  along the **sample** axis, **consistently across modalities** — a microbatch's
  `pixel_values` must correspond to the same samples as its `input_ids`/`labels`,
  or `merge_*`'s per-modality token/feature count check fails. Naive dim-0
  slicing breaks when images-per-sample varies; only the user knows the mapping.
- This mirrors the T006 principle "Cornstarch must not infer": the per-modality
  split is the user's, supplied where they already own batch assembly —
  `collate_fn` (already accepted by `ctx.prepare_dataloader`).

## What you must read
- `cornstarch/distributed/parallelization.py`
  (`ParallelContext.prepare_dataloader` / `wrapped_collate` — DP sampler + CP
  split; `sync_gradients`; the `_default_collate` helper).
- `cornstarch/distributed/context_parallel/splitters.py`
  (`ContextParallelSplitter.compute_offsets`/`split` — applied per microbatch).
- `examples/distributed/pretrain_llm.py` / `pretrain_vlm.py`
  (`_training_step`, the dataloader/`collate_fn` wiring) and
  `examples/distributed/common.py` (`FakeTextDataset`, `causal_lm_criterion`).
- `cornstarch/models/multimodal/execution.py`
  (`ExecutionFuture.execute(inputs=...)` — how a per-microbatch dict is fed).

## Commits
### Commit 1 — `collate_fn` returns a microbatch list; `prepare_dataloader` honors it
- Define the contract: `collate_fn(samples) -> list[dict]`, the list of
  microbatches for one optimizer step. The user owns cross-modality consistency.
- `prepare_dataloader` / `wrapped_collate`:
  - Normalize the collate result to a list: a bare `dict` becomes `[dict]` (the
    single-microbatch case; keeps the default `_default_collate` working).
  - Apply the existing DP-sampler / CP-split transforms **per microbatch** in the
    list (loop the current CP logic over each element).
  - The `DataLoader` then yields a `list[dict]` per iteration.
- Do not split modality tensors anywhere in Cornstarch.

### Commit 2 — non-PP microbatch consumption = gradient accumulation
- Provide the non-PP step path (used when `ctx.uses_pipeline_parallel` is False,
  per T006): iterate the microbatch list, run
  `output_future.execute(inputs=mb)`, `loss = criterion(out, mb) / len(list)`,
  `loss.backward()`; after the loop, `ctx.sync_gradients()` + optimizer step.
- Put this in the distributed examples' `_training_step` (a small local helper).
  Keep the `CornstarchExecutionPlan` build parallelism-agnostic.
- Add an example `collate_fn` that slices a batch into a microbatch list
  (LLM: slice along the sample axis; VLM: keep each sample's pixels with its
  text). Examples stop passing `num_microbatches`/`microbatch_size`.

## Tests
- `prepare_dataloader` with a `collate_fn` returning N microbatches yields a
  `list[dict]` of length N each iteration; CP split (when configured) is applied
  per microbatch. (Can be a small non-distributed/`world_size=1` unit test, plus
  a CP-on gloo check if cheap.)
- Non-PP gradient accumulation: a co-located (T006) run that processes a batch as
  N microbatches accumulates gradients equal (within tolerance) to processing the
  same batch whole. (gloo CPU, `world_size<=2`.)
- `pytest tests` green.

## What you must NOT do
- Do NOT split modality tensors inside Cornstarch (no `ndim`/dim-0 heuristic).
- Do NOT couple `num_microbatches` to anything but the `collate_fn` list length.
- Do NOT change CP/DP numerical behavior.

## Done Condition
- `collate_fn` returning a `list[dict]` is the microbatch interface;
  `prepare_dataloader` yields a list and applies DP/CP per microbatch.
- Non-PP microbatched training does gradient accumulation and matches the
  whole-batch reference.
- `pytest tests` green; branch pushed; PR opened against `feat/T002-parallelism`;
  this file moved to `tasks/done/T007-MICROBATCH-COLLATE.md` with the PR URL.

## HUMAN COMMENTS
- PR targets `feat/T002-parallelism`; auto-merge into it after green (the user
  authorized auto-merging the prerequisites so T008 builds cleanly).

## PR
https://github.com/cornstarch-org/Cornstarch/pull/74 (base: feat/T002-parallelism)
