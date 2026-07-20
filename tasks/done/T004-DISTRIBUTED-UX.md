## Task ID: T004-DISTRIBUTED-UX
## Type: IMPLEMENTATION (interface + examples cleanup; `pytest tests` must stay green)
## Goal: tidy the Option C distributed surface so (a) model materialization has a
single empty/random/HF-checkpoint API that works for both non-parallel and
parallelized models, (b) `ParallelizationPlan.distribute()` is renamed to
`materialize()`, (c) the projector auto-follows its encoder (no manual projector
setup; `from_encoder_and_language_model` removed; `parallelize()` only accepts
Cornstarch models), and (d) the distributed examples read like the
non-distributed `examples/pretrain_vlm.py` (no `train_loop` / `build_plan`
helpers).

Each of the four bullets below is **its own commit** (impl, then any test
update, interleaved logically). Branch from the current
`feat/T003-cp-equivalence` HEAD (it carries the distributed surface this builds
on): `git checkout -b feat/T004-distributed-ux`.

## What you must read
- `cornstarch/models/model_base.py`
  (`set_empty_init`/`set_random_init`/`set_checkpoint_init`, `materialize`,
  `_load_checkpoint_state_dict`, `_materialize_empty` — note the DTensor branch)
- `cornstarch/models/lazy_init.py` (`InitializationPlan` — empty/random/checkpoint)
- `cornstarch/models/multimodal/modeling.py`
  (`CornstarchModalityEncoder`, `from_encoder_and_language_model`, `materialize`,
  `set_*_init` — note it subclasses `nn.Module`, not `CornstarchModelBase`, and
  `materialize` takes no `dtype`)
- `cornstarch/models/multimodal/projector.py` (`CornstarchProjector.materialize`
  — also no `dtype`)
- `cornstarch/distributed/parallelization.py`
  (`ParallelizationPlan.parallelize`/`distribute`, `ParallelContext`)
- `cornstarch/distributed/tensor_parallel/__init__.py` + `plans.py`
  (`apply_tensor_parallel` walks `module._section_names()`)
- `cornstarch/distributed/pipeline_parallel/schedule.py`
  (`NonPipelineParallelSchedule.step` does execute+criterion+backward;
  `OneForwardOneBackwardSchedule.step` does 1F1B)
- `examples/pretrain_vlm.py` (the NON-distributed style to converge toward:
  `build_modality_encoder`, `_training_step`, explicit loop)
- `examples/common.py` (`build_modality_encoder` helper)
- `examples/distributed/{common.py,pretrain_llm.py,pretrain_vlm.py}`
  (`train_loop`, `build_language_model_plan`, inline `build_plan`)
- `tests/distributed/test_parallelism_integration.py` (uses `plan.distribute`)
- `tests/model/test_multimodal_model.py` and
  `tests/model/test_multimodal_materialization.py`
  (use `from_encoder_and_language_model`)

## Reference: how `refactor_distributed` did this (already inspected)
- `InitializationPlan.checkpoint` there has an extra `model_name_or_path` field;
  `set_checkpoint_init(..., model_name_or_path=...)` plus a
  `_download_and_load_safetensors(model_name_or_path, device)` (HfApi +
  `hf_hub_download` + `load_file`, merge all `*.safetensors`) lets a model
  materialize **directly from an HF Hub id**. That is the piece missing on this
  branch (current `checkpoint` only takes `state_dict` / `checkpoint_path`).
- Its distributed example sets the init mode (`set_checkpoint_init(...)` /
  `set_random_init()`) **before** `materialize()`, and the same `materialize()`
  path serves both the plain and the TP/PP-sharded model.

---

## Commit 1 — unified empty / random / HF-checkpoint materialization
**Why:** there is currently no way to materialize a model from an HF Hub id, and
the checkpoint path is unverified for parallelized (DTensor) models. The user
wants one init/materialize API that works for a non-parallelized model AND a
parallelized one.

**Do:**
1. `cornstarch/models/lazy_init.py`: add `model_name_or_path: str | None = None`
   to `InitializationPlan` and to `InitializationPlan.checkpoint(...)`; extend the
   `__post_init__` mutual-exclusion check to cover all three checkpoint sources
   (at most one of `state_dict` / `checkpoint_path` / `model_name_or_path`).
2. `cornstarch/models/model_base.py`:
   - `set_checkpoint_init(self, state_dict=None, checkpoint_path=None,
     model_name_or_path=None)` — thread the new arg through.
   - In `_load_checkpoint_state_dict`, when `model_name_or_path` is set, download
     + merge all `*.safetensors` from the Hub (port
     `_download_and_load_safetensors` as a `@staticmethod`; `huggingface_hub`
     `HfApi().model_info(...).siblings` + `hf_hub_download` + safetensors
     `load_file`), then run the existing `hf_to_cornstarch_state_dict` mapping
     and device/dtype move. Keep `state_dict` and `checkpoint_path` branches.
   - **Parallelized (DTensor) correctness — required.** Verify the `"checkpoint"`
     branch of `materialize()` works when TP recorded DTensor specs on meta
     params. The current `load_state_dict(state_dict, strict=True, assign=True)`
     with a *full* (non-DTensor) tensor will drop the DTensor wrapping / sharding.
     Fix so each rank ends up with its **sharded slice**: e.g. for DTensor params
     run `_materialize_empty` first (its `to_empty` branch preserves the DTensor),
     then copy `distribute_tensor(full_src, p.device_mesh, p.placements)` into
     each DTensor param (mirror the test helper `_copy_full_into` in
     `tests/distributed/test_numerical_equivalence.py`), and `assign`-load the
     plain params as today. Do not regress the non-parallel checkpoint path.
3. Keep `empty` = uninitialized alloc and `random` = `reset_parameters` exactly
   as they are; this commit only adds the Hub source + the DTensor checkpoint fix.

**Test (same commit or immediately after):**
- Non-parallel: `set_checkpoint_init(state_dict=...)` and (network-guarded /
  monkeypatched) `model_name_or_path` round-trip a tiny model.
- Parallel: in `tests/distributed/`, build a tiny LM, `apply_tensor_parallel`,
  `set_checkpoint_init(state_dict=<full ref>)`, `materialize("cpu")`, and assert
  each rank's DTensor shard equals `distribute_tensor(ref)` — i.e. the loss/grad
  matches the non-parallel reference (reuse the gloo harness).
- Gate any real Hub download behind availability so `pytest tests` stays offline-green.

---

## Commit 2 — rename `ParallelizationPlan.distribute()` → `materialize()`
**Why:** consistency with `CornstarchModelBase.materialize()`; `distribute` reads
like a collective.

**Do:**
- Rename the method in `cornstarch/distributed/parallelization.py`. It still
  returns a `ParallelContext`. (Optional: keep a thin `distribute(...)` alias that
  warns and forwards, but prefer a clean rename since this is pre-release —
  decide and state it; default = no alias, update all call sites.)
- Update the class docstring/usage block and `cornstarch/distributed/__init__.py`
  and `parallel_config.py` docstrings (they say "`.distribute()`").
- Update call sites: `tests/distributed/test_parallelism_integration.py:117`,
  `examples/distributed/pretrain_llm.py`, `examples/distributed/pretrain_vlm.py`.
- `grep -rn "\.distribute(" cornstarch tests examples` (excluding
  `distribute_tensor` / `distribute_layers`) must come back clean afterward.

---

## Commit 3 — projector auto-follows the encoder; lock down `parallelize()` input
**Why:** in `examples/distributed/pretrain_vlm.py:107-113` the projector is built,
`set_random_init()`-ed, and `materialize()`-ed by hand AFTER the encoder is
already parallelized+materialized — duplicated and error-prone. Initialization,
materialization (device/dtype), and parallelization of the projector must follow
its encoder automatically, with no separate user calls; and `parallelize()` must
reject anything that isn't a Cornstarch model so users can't pass a bare HF
encoder (which has no projector).

**Do:**
1. `cornstarch/models/multimodal/modeling.py` — make `CornstarchModalityEncoder`
   a first-class lifecycle unit over `(encoder, projector)`:
   - `set_empty_init` / `set_random_init` / **add** `set_checkpoint_init` →
     propagate to BOTH `encoder` and `projector` (so no manual
     `projector.set_random_init()`).
   - `materialize(self, device="cuda", dtype=None)` → **add `dtype`** and
     materialize BOTH encoder and projector in that dtype (encoder already takes
     `dtype`; give `CornstarchProjector.materialize` a `dtype` param, or
     `.to(dtype)` after — projector init stays `reset_parameters`).
   - **Remove** `from_encoder_and_language_model`.
2. Add a package-level builder `build_modality_encoder(encoder, language_model,
   modality, projector_type="linear", **projector_kwargs)` (move the body of the
   removed classmethod) and export it from `cornstarch/models/__init__.py`. Point
   `examples/common.py:build_modality_encoder` at the package one (or delete the
   example shim and import the package function).
3. `cornstarch/distributed/parallelization.py`:
   - `parallelize()` runtime-enforces
     `isinstance(module, (CornstarchModelBase, CornstarchModalityEncoder))`,
     else `TypeError` whose message tells the user to wrap a raw HF encoder with
     `build_modality_encoder(...)`. (This is the misconfiguration guard: a bare
     HF module — or any non-Cornstarch module — is rejected.)
   - In `distribute`/`materialize`, when a registered module is a
     `CornstarchModalityEncoder`, apply TP/CP/PP to its `.encoder` (the apply_*
     helpers need `_section_names()`, which the inner encoder has — the modality
     encoder itself does not), then call `module.materialize(device, dtype)` so
     the projector materializes with it. Projector TP: `apply_tensor_parallel`
     no-ops for an unregistered family, so the projector stays replicated for now
     — acceptable; note it. EP/CP/PP don't apply to the projector.
   - `GradientSynchronizer.register(module)` already walks `module.parameters()`,
     so the modality encoder's projector params are DP-synced automatically; confirm.
4. Update tests that used the classmethod
   (`tests/model/test_multimodal_model.py:362`,
   `tests/model/test_multimodal_materialization.py:33`) to `build_modality_encoder`.

**Decision to record (interface):** prefer the delegation approach above (plan
operates on `modality_encoder.encoder` for the apply_* step) over making
`CornstarchModalityEncoder` subclass `CornstarchModelBase`. Subclassing would
demand an `hf_config` / `hf_model_factory` / state-mapper for a composite that
has none, and would broaden scope; delegation keeps `parallelize()` accepting
both types while the modality encoder owns only its own `(encoder, projector)`
lifecycle. If during implementation the apply_* helpers turn out to need the
modality encoder to *be* a `CornstarchModelBase`, STOP and write a BLOCK section.

---

## Commit 4 — distributed examples mirror the non-distributed style
**Why:** `examples/distributed/*` hide everything behind `train_loop` and
`build_*_plan` in `common.py`, so they look nothing like `examples/pretrain_vlm.py`.
The two should feel almost identical; only the parallelization and schedule
lines may differ.

**Interface analysis (do this first, record findings in the commit message):**
- Non-distributed `examples/pretrain_vlm.py` builds a fresh
  `CornstarchExecutionPlan` per step in `_training_step`, runs
  `output_future.execute(...)`, then `loss.backward()`.
- Distributed builds the plan once and drives it with a `TrainingSchedule`:
  `NonPipelineParallelSchedule.step` already does exactly execute+criterion+
  backward; `OneForwardOneBackwardSchedule.step` does the 1F1B microbatch loop.
- Conclusion: the unifying training-step primitive **already exists** — it is
  `schedule.step(batch, criterion, optimizer, return_loss=True)`. The only
  structural delta the interface forces on a distributed script is: build the
  models the same way, then (i) `init_distributed()`, (ii) `plan.parallelize()` +
  `plan.materialize()` instead of `model.materialize()`, (iii)
  `ctx.prepare_dataloader()` instead of a plain `DataLoader`, (iv) get a schedule
  from `ctx.create_schedule()` and call `schedule.step()` instead of inline
  `execute()`/`backward()`, (v) `ctx.sync_gradients()` before `optimizer.step()`.
  No new interface is required to hit "almost identical"; if implementation
  reveals a gap that still forces divergence beyond these five lines, treat that
  as an interface bug, write a BLOCK section, and propose the fix (e.g. a small
  `ctx.train_step(...)` wrapper) rather than papering over it in the example.

**Do:**
- Delete `train_loop` and `build_language_model_plan` from
  `examples/distributed/common.py` (keep only genuinely shared scaffolding:
  `init_distributed`, `DTYPE`, `FakeTextDataset`, `causal_lm_criterion`, and the
  VLM dataset/collate if shared).
- Rewrite `examples/distributed/pretrain_llm.py` and
  `examples/distributed/pretrain_vlm.py` to follow `examples/pretrain_vlm.py`
  top-to-bottom: build configs → `from_hf_config` → (`build_modality_encoder` for
  VLM) → `set_random_init()` → `plan.parallelize(...)` → `ctx = plan.materialize(...)`
  → `ctx.prepare_dataloader(...)` → build the execution plan inline (a small local
  `_build_plan()`/`_training_step`-shaped function, NOT a common.py helper) →
  `schedule = ctx.create_schedule(...)` → explicit `for batch in loader:` loop
  doing `schedule.step(...)`, `ctx.sync_gradients()`, `optimizer.step()`,
  `optimizer.zero_grad()` (optionally an lr scheduler, like the non-distributed one).
- The VLM script must NOT do any standalone projector setup (Commit 3 makes the
  projector follow the encoder): build the modality encoder with
  `build_modality_encoder`, `parallelize(modality_encoder, ...)`, done.
- Examples are not run by `pytest tests`; smoke-check at least one with
  `torchrun --nproc_per_node=1 ... --steps 1` (or a 2-rank gloo CPU run) and note
  the result in the commit message.

---

## What you must NOT do
- Do NOT import from `cornstarch_old`.
- Do NOT change CP/TP/PP/EP numerical behavior; this is an ergonomics/refactor task.
- Do NOT collapse the four bullets into one commit; one commit per bullet.
- Do NOT make `examples/pretrain_vlm.py` (the non-distributed one) depend on the
  distributed package — convergence is distributed → non-distributed, not the reverse.
- Do NOT introduce a separate checkpoint format; HF-native only.

## On Ambiguity
Per CLAUDE.md: if an interface decision is unclear (e.g. whether the modality
encoder must subclass `CornstarchModelBase`, or whether to keep a deprecated
`distribute` alias), write your interpretation under a BLOCK section here and stop.

## Done Condition
- `set_checkpoint_init(model_name_or_path=...)` materializes a model from an HF
  Hub id, and checkpoint init produces correctly-sharded weights on a TP model
  (new distributed test green).
- `ParallelizationPlan.materialize()` replaces `distribute()`; no stale
  `.distribute(` call sites remain.
- `CornstarchModalityEncoder.from_encoder_and_language_model` is gone;
  `build_modality_encoder` is the package builder; `parallelize()` rejects
  non-Cornstarch modules; the VLM example has zero standalone projector calls.
- `examples/distributed/*` contain no `train_loop` / `build_*_plan` helpers and
  read like `examples/pretrain_vlm.py` apart from the five distributed lines.
- `pytest tests` is green; branch pushed; PR opened against
  `feat/T003-cp-equivalence` with the four commits; this file moved to
  `tasks/done/T004-DISTRIBUTED-UX.md` with the PR URL.

---

## PR
https://github.com/cornstarch-org/Cornstarch/pull/71
