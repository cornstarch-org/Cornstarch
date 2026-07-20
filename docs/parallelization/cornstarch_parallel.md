# Composable Cornstarch parallelism

`ParallelizationPlan` assigns the complete distributed world to per-module
five-axis grids. Each module receives its own `ParallelConfig`, so a vision
encoder and language model can use different TP, CP, PP, or EP degrees while
sharing the same DP replica count.

## Build and materialize a plan

```python
import torch
from cornstarch.distributed import (
    HeadTailContextParallelSplitter,
    ParallelConfig,
    ParallelizationPlan,
)

plan = ParallelizationPlan()
plan.parallelize(
    vision,
    ParallelConfig(
        data_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        tensor_parallel_size=2,
    ),
)
plan.parallelize(
    language_model,
    ParallelConfig(
        data_parallel_size=2,
        pipeline_parallel_size=2,
        context_parallel_size=2,
        tensor_parallel_size=2,
        expert_parallel_size=2,
        context_parallel_splitter=HeadTailContextParallelSplitter(),
    ),
)

context = plan.materialize("cuda", dtype=torch.bfloat16)
```

All modules must agree on whether PP is enabled. With PP disabled they are
co-located and must use equal ranks per replica. With PP enabled they occupy
disjoint stage ranges. The plan verifies that `global_ranks` contains every
world rank exactly once and that each explicit DP size matches the inferred
replica count.

## Why materialization order matters

Models begin on `meta`. The plan applies TP, CP, and PP before allocating model
storage, materializes the local model partition, then applies EP to concrete
batched expert tensors. Users should not manually reorder those primitives:
doing so can allocate full weights, lose DTensor placements, or slice empty
expert storage.

## Data and context parallelism

DP and CP are data-related dimensions and do not belong in family-specific model
forwards. `context.prepare_dataloader()` installs a DP sampler and a CP collate
transform. The transform preserves global positions and shifted labels before
slicing, records cross-mesh ownership, and supplies recurrent-run metadata when
a linear-attention token mixer needs it.

Available ownership layouts are:

- `UniformContextParallelSplitter`: one contiguous region per rank;
- `HeadTailContextParallelSplitter`: mirrored head and tail regions, balancing
  causal attention work;
- `MakespanMinContextParallelSplitter`: work-aware ownership for multimodal
  attention patterns.

The old `ZigzagContextParallelSplitter` spelling is only a deprecated alias for
head-tail ownership.

## Pipeline, tensor, and expert parallelism

- **TP** records DTensor projection placements on each unified repeated layer.
  Unsupported converted structures fail explicitly instead of silently running
  replicated weights.
- **PP** slices the repeated `ModuleList` and derives stage behavior from the
  model's forward spec and the user's execution DAG.
- **EP** shards batched expert tensors and installs differentiable all-to-all
  token dispatch. Expert shards stay fixed while orthogonal CP/DP synchronization
  combines corresponding gradients.

## Cross-module boundaries

For different modality and language-model grids, Cornstarch constructs seam
groups in deterministic world order. A variable-split autograd all-to-all sends
each projected feature row only to the LLM CP rank that owns its placeholder.
Backward performs the transposed exchange, so neither direction gathers or
broadcasts the full modality sequence.

## Training loop

```python
loader = context.prepare_dataloader(dataset, batch_size=8, collate_fn=collate)
schedule = context.create_schedule(graph, language_output)

for microbatches in loader:
    optimizer.zero_grad()
    result = schedule.step(microbatches, criterion, return_loss=True)
    context.sync_gradients()
    optimizer.step()
```

Without PP, execute the output future directly for each microbatch and call
`context.sync_gradients()` before the optimizer step.
