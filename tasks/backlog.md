# Backlog — deferred work

Items explicitly deferred during T002-PARALLELISM (see HUMAN DECISION in
specs/drafts/parallelism_interface_options.md). Resume after the core parallelism
interface lands.

## B1 — Distributed checkpointing (sharded save/load)
- **Deferred from:** T002-PARALLELISM (human decision: "ignore all checkpointing
  feature for now; focus on parallelism").
- **Scope:**
  - `gather_hf_state_dict(model, mesh)` shim that reconstructs the full HF-native
    state dict on rank 0 (all-gather TP DTensor shards, gather PP `ModuleList`
    slices, un-stack EP `_BatchedExperts` experts), then reuse the existing
    `save_pretrained` unchanged (no separate checkpoint format).
  - **Sharded lazy load (required):** each rank stages only its own shard via the
    existing `InitializationPlan` / `set_checkpoint_init` path — slice the full HF
    tensor down to the local shard in `_load_checkpoint_state_dict` before
    `load_state_dict(assign=True)`, so no rank materializes the whole model.
- **Constraint:** stays HF-native; no new on-disk format.

## B2 — ZeRO / optimizer-state sharding across DP
- **Deferred from:** T002-PARALLELISM (human decision: "keep it in todo list; we
  will address it later").
- **Scope:** shard optimizer state across the DP group (current design replicates
  optimizer state and only all-reduces gradients). Additive layer over the chosen
  interface; should not change the Option B/C user-facing surface.

## B3 — Linear (gated-delta-net) attention as a pipeline boundary stage
- **Found during:** T002-PARALLELISM (EP+PP composition testing).
- **Symptom:** when a Qwen3.5-MoE `linear_attention` layer is the *only* layer
  on the last PP stage, the stage forward produces a NaN loss. The same layer is
  numerically fine in the non-PP model and under EP/TP/DP without PP. Full-
  attention layers pipeline correctly.
- **Scope:** investigate the gated-delta-net forward when it runs as an isolated
  PP boundary stage (likely conv/recurrent-state or mask handling under the
  stage-aware forward spec). Until fixed, PP over models with linear-attention
  layers is unsupported; `test_parallelism_integration` uses an all-full-
  attention MoE config for EP+PP, and `test_expert_parallel_qwen` covers
  linear-attention EP without PP.

## B4 — Causal-LM context-parallel numerical equivalence
- **Deferred from:** T003-CP-EQUIVALENCE.
- **Reason:** the CP all-gather flash-attention kernel
  (`cornstarch/distributed/context_parallel/attention.py`) hardcodes
  `causal=False` (`_flash_attn_forward(..., causal=False)`) and splits the
  sequence uniformly, so it cannot reproduce a causal language model under CP.
  T003 therefore tests CP equivalence only at the **attention-function level**
  against a non-causal full-sequence SDPA reference
  (`tests/distributed/test_numerical_equivalence.py::TestContextParallelEquivalence`).
- **Scope to resume:** add causal masking to the CP kernel and a load-balanced
  causal split (zigzag / makespan-style assignment so each rank gets a balanced
  share of the triangular work), then assert a full causal-LM forward+backward
  CP equivalence vs the single-GPU reference.
