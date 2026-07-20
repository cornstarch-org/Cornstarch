# Core architecture

Cornstarch is not a monolithic multimodal model wrapper or a parallelism plugin
that replaces the user's training loop. It is a small set of stable model,
graph, and distribution contracts. Model-family details terminate at conversion;
modality composition remains user-defined; parallel dimensions stay composable.

## 1. Lazy initialization and rank-local ownership

`from_hf_config()` constructs modules under `torch.device("meta")`. Meta tensors
describe names, shapes, dtypes, and module topology without allocating model
storage. A checkpoint source or random-initialization policy is recorded in an
`InitializationPlan`; it is not executed by the converter.

The order inside `ParallelizationPlan.materialize()` is deliberate:

1. TP records DTensor placements on meta parameters.
2. CP injects the appropriate attention or recurrent token-mixer function.
3. PP slices the unified repeated-layer stack for the local stage.
4. `materialize()` allocates or assigns the tensors owned by this rank.
5. EP slices concrete batched expert tensors and installs token routing.

Changing that order can allocate a full model before sharding, discard DTensor
placement metadata, or try to slice expert tensors that have no storage. The
ordering is therefore a lifecycle invariant, not an implementation detail.

## 2. Unified module representation

Hugging Face families expose different root objects and forward methods, but
Cornstarch parallelism sees only two transformer roots:

- `CornstarchLanguageModel` with `pre_decoder`, `decoder_layers`, and
  `post_decoder` sections;
- `CornstarchEncoder` with `pre_encoder`, `encoder_layers`, and `post_encoder`
  sections.

Every repeated section is an `nn.ModuleList`. A family converter may reuse Hugging
Face leaf modules and supplies a `TransformerForwardSpec` plus a checkpoint
prefix map. The shared Cornstarch loop still owns repeated-layer execution, and
the shared base owns materialization, offload, compilation, and checkpoint
translation. Consequently, distributed code walks `_section_names()` instead of
selecting a model-family policy or copied forward implementation.

Adding a model family means adding a converter. An unsupported config is rejected;
`model_kind="language"` or `"vision"` is only an assertion and never silently
chooses a Llama, CLIP, or Whisper converter.

## 3. User-specified multimodal DAGs

Cornstarch does not impose one `MultimodalModel.forward()`. Users record nodes in
a `CornstarchExecutionPlan`. Each node returns an `ExecutionFuture`; passing that
future to another node creates a dependency edge.

```python
graph = CornstarchExecutionPlan()
vision = graph.run_modality_encoder(vision_module, pixel_values=pixels)
merged = graph.merge_modality_encoder_outputs(
    language_model=language_model,
    input_ids=input_ids,
    labels=labels,
    modality_token_ids={"vision": image_token_id},
    encoder_outputs={"vision": vision},
)
output = graph.run_language_model(language_model, inputs=merged)
```

The plan validates and topologically sorts the dependency closure. Executing an
intermediate future runs only its ancestors, which keeps debugging and custom
multimodal flows possible. The PP schedule consumes the same DAG rather than a
second parallel-only model forward.

## 4. Per-module, composable parallelization

Each module registered with `ParallelizationPlan` has its own `ParallelConfig`.
The plan checks that `global_ranks` contains the whole distributed world exactly
once, computes the DP replica count, and assigns module-specific PP/CP/TP/EP
grids. Modules may be co-located or disaggregated into pipeline stages, but all
participants create process groups in the same deterministic order.

When a modality encoder grid and language-model grid differ, the pipeline seam
uses a differentiable variable-split all-to-all. It transfers only projected
feature rows owned by the destination CP rank and routes gradients back to their
source, without gathering full modality sequences.

The dimensions remain independent:

| Dimension | Owns | Integration point |
| --- | --- | --- |
| DP | samples and replica gradient averaging | sampler and training context |
| CP | token ownership and partial-gradient sums | collate transform and injected token mixer |
| TP | hidden/head parameter shards | unified repeated-layer structure |
| PP | repeated-layer stage ownership | unified sections and execution DAG |
| EP | expert parameter shards and token routing | batched expert modules |

## 5. Data parallelism stays out of model implementations

DP and CP must not become branches inside a model-family forward. The
`ParallelContext` wraps collation to select the DP sample shard, preserve global
positions and shifted labels, calculate CP ownership, and slice sequence fields.
For CP, `apply_context_parallel()` injects a group-bound token-mixer callable
during materialization; the reused layer receives rank-local inputs and opaque
sequence metadata.

TP, PP, and EP necessarily alter parameter or layer ownership, but they operate
once over the Cornstarch section contract. Model-family conversion is allowed to
describe leaf paths and forward semantics; distributed execution must not grow a
parallel policy or copied forward for every Hugging Face family.
