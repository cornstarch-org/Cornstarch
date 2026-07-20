# PyTorch DDP and FSDP

Cornstarch's supported distributed path is `ParallelizationPlan`. Its DP axis
adds distributed sampling and gradient synchronization while remaining
composable with module-specific PP, CP, TP, and EP grids.

## DDP

For a single co-located Cornstarch module, ordinary PyTorch DDP can be used after
materialization. A user-composed multimodal DAG has no single required root
`nn.Module`, however, so wrapping independent modules separately also requires
the application to coordinate unused parameters and execution order. Prefer the
native DP axis when using multiple modules or any other Cornstarch parallel
dimension.

## FSDP

FSDP is not currently part of the supported Cornstarch composition contract.
Cornstarch TP already uses DTensor parameter placement, PP owns explicit layer
stages, and EP owns expert shards; adding FSDP without an explicit mesh and
checkpoint design can overlap parameter ownership incorrectly. Do not combine
FSDP with a `ParallelizationPlan` unless that combination gains dedicated
materialization, optimizer-state, and numerical-equivalence coverage.

For replica scaling today, configure `data_parallel_size` and call
`context.sync_gradients()` after backward and before the optimizer step.
