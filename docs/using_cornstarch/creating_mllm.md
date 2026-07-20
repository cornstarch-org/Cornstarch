# Creating a multimodal model

Cornstarch represents multimodal models as independently owned modules connected
by an execution DAG. There is no required multimodal root class and no borrowed
Hugging Face multimodal `forward()`.

## Convert the component models

Converters build unified Cornstarch roots on the `meta` device. Construction
does not allocate the model weights.

```python
from transformers import AutoConfig
from cornstarch.models import from_hf_config

text_config = AutoConfig.from_pretrained("org/text-model")
vision_config = AutoConfig.from_pretrained("org/vision-model")

language_model = from_hf_config(text_config.text_config)
vision_encoder = from_hf_config(vision_config.vision_config)
```

Every supported text config returns `CornstarchLanguageModel`; every supported
vision or audio config returns `CornstarchEncoder`. Family-specific converters
only arrange leaf modules, forward hooks, and Hugging Face checkpoint prefixes.

## Add the projector

`build_modality_encoder()` combines a unified encoder with a projector whose
output width matches the language model.

```python
from cornstarch.models import build_modality_encoder

vision = build_modality_encoder(
    vision_encoder,
    language_model,
    modality="vision",
    projector_type="mlp",
)
```

The returned `CornstarchModalityEncoder` owns the encoder and projector, but it
does not decide where its result flows. That is the DAG's responsibility.

## Construct the DAG with futures

Every plan method records a node and returns an `ExecutionFuture`. Passing a
future as another node's input creates an edge.

```python
from cornstarch.models import CornstarchExecutionPlan, ExecutionFuture

graph = CornstarchExecutionPlan()
vision_features = graph.run_modality_encoder(
    vision,
    pixel_values=ExecutionFuture("pixel_values"),
)
merged = graph.merge_modality_encoder_outputs(
    language_model=language_model,
    input_ids=ExecutionFuture("input_ids"),
    labels=ExecutionFuture("labels"),
    modality_token_ids={"vision": image_token_id},
    encoder_outputs={"vision": vision_features},
)
language_output = graph.run_language_model(language_model, inputs=merged)
```

`language_output.execute(batch)` runs only the dependency closure needed for
that future. `graph.execute(batch)` runs the full graph. The same graph is read
by pipeline scheduling, so serial and pipeline execution do not maintain
separate multimodal forwards.

## Inspect and validate

```python
graph.validate()
print(graph.describe())
print(graph.to_mermaid())
```

Validation rejects cycles, duplicate output names, and unresolved default merge
inputs before execution begins.
