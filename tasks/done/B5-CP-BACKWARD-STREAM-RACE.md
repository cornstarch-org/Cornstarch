## Task ID: B5-CP-BACKWARD-STREAM-RACE
## Type: BUGFIX (`pytest tests` must stay green)
## Goal: fix the side-stream race in
`ContextParallelFlashAttention.backward` so the per-head-group ``dkv`` is only
cloned **after** its side-stream ``reduce_scatter`` has landed, then drop the
test-only ``_serialize_cp_stream`` workaround introduced in T003.

## Motivation (from backlog B5)
In `cornstarch/distributed/context_parallel/attention.py`, the backward overlaps
each head group's gradient ``reduce_scatter`` on a dedicated side CUDA stream:

```
with torch.cuda.stream(stream):
    dist.reduce_scatter(dkv[0], ..., async_op=True)
    dist.reduce_scatter(dkv[1], ..., async_op=True)

dqs.append(dq.clone())
dks.append(dkv[0].clone())   # <-- default stream, races the side-stream writes
dvs.append(dkv[1].clone())
...
torch.cuda.current_stream().wait_stream(stream)   # <-- only sync, AFTER the loop
```

``dkv`` is a ``torch.empty`` written by the side-stream ``reduce_scatter``, but
``dkv[0].clone()`` / ``dkv[1].clone()`` run on the **default** stream before that
side-stream copy is guaranteed to have completed (the only ``wait_stream`` is
after the whole loop). The clone can read uninitialized/partial memory →
intermittent corrupted ``dk``/``dv`` or NaN. (``dq`` is computed by
``_flash_attn_backward`` on the default stream and is not part of the race.)

T003 worked around this in the test only (`_serialize_cp_stream` pins the kernel
overlap stream to the current stream so collectives serialize with compute). The
kernel itself is still wrong; this task fixes it and removes the workaround.

## What you must read
- `cornstarch/distributed/context_parallel/attention.py`
  - `_allgather_kv` (the side-stream all-gather; note it records a **per-head-group
    event** on the side stream and the consumer does
    `current_stream().wait_event(per_head_events[gi])` before use — mirror this
    pattern for the backward reduce-scatter).
  - `ContextParallelFlashAttention.forward` (per-head-group event sync as the
    reference correct pattern) and `.backward` (the racy loop, lines ~194–238).
  - `_get_stream` / the class-level `_stream` (how the side stream is obtained;
    the test overrides it).
- `tests/distributed/test_numerical_equivalence.py`
  - `_serialize_cp_stream` (the workaround to delete) and its two call sites
    (`TestContextParallelEquivalence` forward + backward tests, ~lines 391, 481).
  - Backward parity must still hold at the existing tolerance (≈5e-3) **without**
    the workaround.
- `tasks/backlog.md` B5 (this item); do not conflate with B4 (separate).

## Proposed approach (refine during implementation; BLOCK on a real interface
## decision per CLAUDE.md)

### Commit 1 — sync the reduce-scatter before cloning ``dkv`` (kernel fix)
Make the backward wait on the side stream's reduce-scatter for a head group
before that group's ``dkv`` is cloned, preserving as much overlap as possible.
Preferred: mirror the forward's per-group event pattern —
- inside `with torch.cuda.stream(stream):`, after the two `reduce_scatter`
  calls, record an event on the side stream (`evt = torch.cuda.Event();
  evt.record(stream)`);
- before `dks.append(dkv[0].clone())` / `dvs.append(dkv[1].clone())`, do
  `torch.cuda.current_stream().wait_event(evt)`.

This makes each group's clone observe its own completed reduce-scatter while the
side stream can still run ahead, so the overlap optimization is retained. (The
final `wait_stream(stream)` after the loop may stay as a backstop.) Acceptable
alternatives if the event path is awkward: a per-iteration
`current_stream().wait_stream(stream)` before the clones, or have
`reduce_scatter` write into a freshly-allocated (non-aliased) output and capture
the `async_op` work handle and `.wait()` it. State whichever you choose and why;
**do not change the kernel math** (numbers must match the pre-fix serialized
result).

### Commit 2 — drop the test workaround
- Remove `_serialize_cp_stream` and its two call sites from
  `tests/distributed/test_numerical_equivalence.py`; the CP forward/backward
  equivalence tests should run the kernel on its real side stream and still pass
  at the current tolerance.
- Keep the tests CUDA-/flash-attn-guarded exactly as they are
  (`@unittest.skipUnless(...)`), so `pytest tests` stays green where the kernel
  cannot run.

## Verification
- This kernel is **CUDA + flash-attn only**; the equivalence tests are guarded
  and skip without a GPU. Run them on a CUDA host:
  `pytest tests/distributed/test_numerical_equivalence.py -k ContextParallel`
  and confirm forward+backward parity holds (≈5e-3) with the workaround removed.
  If no GPU is available in this environment, state that the kernel fix is
  verified by code review + the unchanged-math argument, that the guarded tests
  still pass-by-skip in `pytest tests`, and note the GPU run as the remaining
  manual check in the PR/done file.
- `pytest tests` stays green (offline).
- `ruff check cornstarch/` clean.

## What you must NOT do
- Do NOT change the CP attention math or numerical results — this is purely a
  stream-ordering correctness fix.
- Do NOT address B4 (causal-LM CP equivalence / causal masking + load-balanced
  split) here; B5 is only the backward stream race.
- Do NOT import from `cornstarch_old`.

## On Ambiguity
If event-vs-`wait_stream` ordering interacts with the `async_op=True` work handle
in a way that is unclear (e.g. whether a recorded event captures an `async_op`
collective's completion on the side stream for the backend in use), record the
interpretation here under a BLOCK section and stop rather than guessing.

## Done Condition
- The backward no longer clones ``dkv`` before its reduce-scatter completes; the
  ordering is enforced in the kernel (event/`wait_stream`), math unchanged.
- `_serialize_cp_stream` and its call sites are gone; the guarded CP equivalence
  tests pass on CUDA without it (or the GPU run is noted as the remaining manual
  check if no GPU is available here).
- `pytest tests` green; `ruff` clean.
- Branch pushed; PR opened against the appropriate base; B5 removed from
  `tasks/backlog.md`; this file moved to `tasks/done/B5-CP-BACKWARD-STREAM-RACE.md`
  with the PR URL.

## HUMAN COMMENTS
- (PR base branch? Prior parallelism work merged into `feat/T002-parallelism`;
  confirm whether this fix targets the same.)

## PR
https://github.com/cornstarch-org/Cornstarch/pull/77 (base: feat/T002-parallelism; left open for human review)

## RESOLUTION NOTES
- Base branch (HUMAN COMMENT): targeted `feat/T002-parallelism`, consistent with
  prior parallelism PRs (T005 #?, T006 #75, T008) which all merge there.
- Removing `_serialize_cp_stream` exposed TWO races, not one. The spec described
  only the consumer race (clone-before-reduce-scatter). The producer race
  (`_flash_attn_backward` writes `dgkv` on the default stream, side-stream
  `reduce_scatter` consumed it unsynced) also had to be fixed —
  `stream.wait_stream(current_stream())` before the reduce-scatter — or the
  CP equivalence test failed intermittently with a graded ~6% dk/dv mismatch.
- No BLOCK needed on the event-vs-async_op ambiguity: the forward already uses
  `evt.record(stream)` after an `async_op=True` collective as the sanctioned
  pattern, and the gloo bridge enqueues its data-movement GPU ops on the current
  (side) stream, so a side-stream event captures their completion.
- GPU available in this environment: CP equivalence tests ran for real and pass
  (3 repeated runs, 5e-3) with the workaround removed; full `pytest tests` 170 passed.
