<div align="center">

![Cornstarch Logo](https://cornstarch-org.github.io/assets/images/cornstarch.svg)

# Cornstarch

Build, compose, and parallelize multimodal language models.

</div>

Cornstarch turns supported Hugging Face language and encoder configs into a
small set of parallelization-aware module representations. Models are created
on the `meta` device, independently connected as a user-defined execution DAG,
and then materialized according to a per-module DP/PP/CP/TP/EP plan.

Its design centers on five properties:

- **Lazy initialization:** parallelism changes parameter ownership before real
  storage is allocated or checkpoint values are assigned.
- **Unified modules:** every supported LLM becomes
  `CornstarchLanguageModel`; every supported vision or audio backbone becomes
  `CornstarchEncoder`. Family converters translate leaf layouts but do not
  create family-specific runtime wrappers.
- **User-defined DAGs:** `ExecutionFuture` values connect modality encoders,
  merges, and language-model calls without a fixed multimodal root model.
- **Composable parallelism:** each module receives its own `ParallelConfig`.
  Cornstarch assigns the whole world to module grids and routes activations at
  grid boundaries.
- **Clean data/model separation:** DP sampling and CP token ownership live in
  the dataloader/training context. TP, PP, and EP operate through the unified
  model structure; CP only injects token-mixer behavior during initialization.

```python
import torch
from cornstarch.distributed import ParallelConfig, ParallelizationPlan
from cornstarch.models import from_hf_config

language_model = from_hf_config(hf_text_config)
language_model.set_checkpoint_init(model_name_or_path="org/model")

plan = ParallelizationPlan()
plan.parallelize(
    language_model,
    ParallelConfig(
        data_parallel_size=2,
        pipeline_parallel_size=2,
        context_parallel_size=2,
        tensor_parallel_size=2,
        expert_parallel_size=2,
    ),
)

# TP/CP/PP are applied while parameters are meta; only this rank's model
# partition is materialized, followed by EP over concrete expert tensors.
context = plan.materialize("cuda", dtype=torch.bfloat16)
```

See [Core architecture](docs/architecture.md), the
[documentation](https://cornstarch-org.github.io), and the runnable examples in
[`examples/`](examples/).

## Research papers

- [Cornstarch: Distributed Multimodal Training Must Be Multimodality-Aware](https://arxiv.org/abs/2503.11367)
- [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates](https://arxiv.org/abs/2309.08125)

## Contact

Insu Jang (insujang@umich.edu)
