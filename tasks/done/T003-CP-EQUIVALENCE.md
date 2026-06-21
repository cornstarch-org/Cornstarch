## Status: DONE — PR https://github.com/cornstarch-org/Cornstarch/pull/70
## (base: feat/T002-parallelism; T003 builds on T002's CP kernel + test file)

## Task ID: T003-CP-EQUIVALENCE
## Type: IMPLEMENTATION (single-GPU CP numerical-equivalence test; tests must be green)
## Goal: replace the skipped (>=2 GPU) context-parallel numerical-equivalence test
with a single-GPU test, following the legacy single-GPU CP testing approach.

In T002-PARALLELISM the CP numerical-equivalence test
(`tests/distributed/test_numerical_equivalence.py::TestContextParallelEquivalence`)
was guarded to require >=2 CUDA devices and therefore skips on this 1-GPU box.
The legacy suite tested CP on a single GPU and we should do the same.

## Feasibility (already verified — do not re-litigate)
- **It works on one GPU.** `GlooDistributedTestBase` spawns `world_size`
  processes and sets `torch.cuda.set_device(rank % device_count())`; with one
  GPU every rank shares `cuda:0`. The collectives run on the **gloo** backend
  (CPU) via the `gloo_utils` patches, which bridge CUDA tensors through CPU
  (`all_gather_gloo` / `reduce_scatter_gloo` copy to CPU, communicate, copy
  back), while flash-attention computes on the shared GPU.
- **Confirmed empirically** against the *new*
  `cornstarch.distributed.context_parallel.attention.ContextParallelFlashAttention`:
  a 2-rank gloo run on a single GPU passed forward **and** backward parity at
  `rtol=atol=5e-3` (bf16) vs a full-sequence reference. The exact shape of the
  passing probe is the legacy `TestFlashAttentionContextParallelismClass`
  (`tests_old/test_shardformer/test_context_parallel.py`, lines 401-493).
- **Scope limit (important).** The new CP kernel hardcodes `causal=False`
  (`attention.py`: `_flash_attn_forward(..., causal=False)`). So CP equivalence
  is testable at the **attention-function level** against a *non-causal*
  reference (`scaled_dot_product_attention(..., is_causal=False)`), exactly as
  the legacy flash-attn CP test did. A full **causal language-model** CP
  equivalence is NOT feasible with the current kernel (it would need causal
  masking + a load-balanced split); that is a separate, larger item — see
  Backlog note below, do NOT attempt it here.

## What you must read
- tests_old/test_shardformer/test_context_parallel.py
  (`TestFlashAttentionContextParallelismClass`, the single-GPU pattern to follow)
- cornstarch/distributed/context_parallel/attention.py
  (`ContextParallelFlashAttention`, `context_parallel_flash_attention`,
  `_allgather_kv` — note `causal=False`, `softmax_scale = d ** -0.5`,
  `heads_stride`)
- tests/distributed/distributed_base.py (GlooDistributedTestBase, the CUDA
  device-per-rank setup and the dynamo-disable already added)
- tests/distributed/gloo_utils.py (all_gather_gloo / reduce_scatter_gloo —
  confirm they handle the attention's `async_op=True` calls and CUDA tensors)
- tests/distributed/test_numerical_equivalence.py
  (the current skipped `TestContextParallelEquivalence` to replace)

## What you must produce
1. **A single-GPU CP attention numerical-equivalence test** in
   `tests/distributed/` (either a new `test_context_parallel_attention.py` or by
   replacing `TestContextParallelEquivalence` in
   `test_numerical_equivalence.py`). It must:
   - subclass `GlooDistributedTestBase` with `world_size = 2`;
   - `@unittest.skipUnless(torch.cuda.is_available(), ...)` (single GPU is
     enough — do NOT require >=2 devices);
   - build full `q, k, v` of shape `(batch, seq, heads, dim)` on `cuda` in
     bf16 (use `dim=64`, `heads=8`, which flash-attn supports);
   - compute the reference with non-causal SDPA over the full sequence;
   - chunk `q, k, v` along the sequence dim per rank, call
     `ContextParallelFlashAttention.apply(local_q, local_k, local_v,
     dist.GroupMember.WORLD)` (use the WORLD group as the CP group — no
     `DeviceMesh`, which avoids the cuda-device-type + gloo-backend mismatch);
   - assert `cp_out` matches `chunk(ref_out)[rank]` and that `cp_dq/dk/dv`
     match `chunk(ref_d*)[rank]` (forward + backward), using `rtol=atol=5e-3`
     (bf16 CP all-gather flash-attn accumulates differently than a single
     matmul; 1e-3 is too tight here — verified). Justify the tolerance in a
     comment, referencing that the broader project target is 1e-3 but CP
     bf16 flash-attn needs 5e-3, matching the legacy test.
   - parametrize over a few `(batch_size, seq_len)` like the legacy test
     (e.g. bs in {1, 2}, seq in {128, 256, 1024}); keep seq divisible by
     `world_size`.
2. **Remove the >=2-GPU skip** path: delete or replace the old
   `TestContextParallelEquivalence` so CP equivalence actually runs on this box.
   If kept in `test_numerical_equivalence.py`, drop the
   `skipUnless(device_count() >= 2)` and the `DeviceMesh(device_type="cuda")`
   usage in favor of the WORLD-group attention-level test above.
3. **(Optional, only if cheap)** also exercise the HF dispatch wrapper
   `context_parallel_flash_attention(module=None, query, key, value, cp_group=...)`
   (the `(b, h, s, d)` -> `(b, s, h, d)` transpose path) for one shape, to cover
   the registered-attention entry point, not just the raw autograd Function.
4. **Backlog note**: append an item to `tasks/backlog.md` recording that
   full **causal-LM** CP numerical equivalence is deferred because the CP
   flash-attention kernel is non-causal (`causal=False`) and lacks
   load-balanced causal splitting; resuming it means adding causal support
   (and likely the zigzag/makespan split) to the CP kernel.

## What you must NOT do
- Do NOT require >=2 GPUs; the test must run on a single GPU via 2 gloo ranks.
- Do NOT try to make a full causal language model match under CP — the kernel
  is non-causal; that is out of scope and goes to the backlog.
- Do NOT change the CP kernel's behavior (no adding causal support in this task).
- Do NOT import from `cornstarch_old`; port the *pattern*, not the legacy code
  (the legacy `ContextParallelFlashAttention` lives in `cornstarch_old`).
- Do NOT add a `DeviceMesh` with `device_type="cuda"` under the gloo backend in
  the test; use `dist.GroupMember.WORLD` as the CP group.

## On Ambiguity
Per CLAUDE.md: write your interpretation to this file under a BLOCK section and
stop. Do not guess on interface decisions.

## RESOLVED (2026-06-21): backward `dk` instability was a kernel race, not bf16

First diagnosis was that backward `dk` failed `5e-3` by 3–10× due to bf16
cross-rank gradient reduction, and the user approved loosening backward to
`5e-2`. Deeper investigation **disproved that**: the real cause is a side-stream
race in `ContextParallelFlashAttention.backward`. The per-head `dkv` (a
`torch.empty`) is `.clone()`d on the default stream right after an
`async_op=True` `reduce_scatter` on the kernel's side stream, with the only
`wait_stream` *after* the loop — so the clone intermittently reads
uninitialized memory (corrupted `dk`, occasional NaN).

Evidence: pinning the kernel's overlap stream to the current stream (serializing
the collectives — no math change, the side stream is a pure perf optimization)
makes worst-case backward grad diff **≤ 0.00098 with zero NaNs over 80+ runs**
across all shapes. So backward parity holds at the task's intended **5e-3**.

Final implementation:
- `_serialize_cp_stream()` pins `ContextParallelFlashAttention._stream` to the
  current stream (test-only; documented). This removes the race.
- Forward **and** backward asserted at `rtol=atol=5e-3` (the spec value), not the
  approved `5e-2` — the `5e-2` is unnecessary once the race is gone, and `5e-3`
  matches the Done Condition and the legacy test exactly.
- The latent kernel race is recorded as **B5** in `tasks/backlog.md`.

## Done Condition
- `tests/distributed/` contains a CP numerical-equivalence test that runs (not
  skips) on a single GPU and asserts forward + backward parity vs a non-causal
  full-sequence reference at `rtol=atol=5e-3` bf16.
- The previous >=2-GPU-gated `TestContextParallelEquivalence` no longer skips on
  a 1-GPU machine (replaced or removed).
- `pytest tests/distributed/test_numerical_equivalence.py` (or the new test
  file) is green on this box; `pytest tests` remains green.
- The deferred causal-LM CP equivalence is recorded in `tasks/backlog.md`.
