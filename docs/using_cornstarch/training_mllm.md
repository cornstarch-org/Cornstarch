# Training a multimodal model

Cornstarch modules remain ordinary `torch.nn.Module` objects and their parameters
remain enumerable by stock PyTorch optimizers. The execution DAG replaces only
the assumption that a multimodal model must have one fixed root `forward()`.

## Unparallelized or co-located execution

```python
optimizer.zero_grad()
for microbatch in microbatches:
    output = language_output.execute(microbatch)
    (output.loss / len(microbatches)).backward()
optimizer.step()
```

## Full, frozen, and LoRA fine-tuning

Use the same per-module configuration before local or distributed
materialization. PEFT adapter injection is deferred automatically for lazy
Cornstarch models, so checkpoint loading and parallel rank ownership are already
resolved when the adapter is created.

```python
from peft import LoraConfig
from cornstarch.models import configure_finetuning

lora = LoraConfig(
    target_modules="all-linear",
    r=8,
    lora_alpha=16,
)

configure_finetuning(vision_module, "lora", lora_config=lora)
configure_finetuning(language_model, "frozen")

# Local:
vision_module.materialize(device)
language_model.materialize(device)

# Distributed uses the same configuration calls above:
# context = parallelization_plan.materialize(device)
```

Each encoder and the language model accepts `full`, `frozen`, or `lora`
independently. Passing a `CornstarchModalityEncoder` configures only its encoder;
the modality projector remains trainable unless the caller freezes it explicitly.

| Encoder mode | LLM mode | Result |
| --- | --- | --- |
| `lora` | `full` | encoder adapters plus full LLM fine-tuning |
| `lora` | `frozen` | encoder adapters with a completely frozen LLM |
| `full` | `lora` | full encoder fine-tuning plus LLM adapters |
| `frozen` | `lora` | frozen encoder plus LLM adapters |

In `lora` mode, PEFT freezes the base module and exposes only adapter parameters
to the optimizer. In `frozen` mode, the whole selected base module has
`requires_grad=False`. Optimizers remain ordinary PyTorch optimizers and should
continue to select parameters with `requires_grad=True`.

## Pipeline execution

When modules are disaggregated into PP stages, the `ParallelContext` derives a
rank-local 1F1B program from the same DAG.

```python
schedule = context.create_schedule(graph, language_output)

for microbatches in loader:
    optimizer.zero_grad()
    result = schedule.step(
        microbatches,
        criterion=lambda output, batch: output.loss,
        return_loss=True,
    )
    context.sync_gradients()  # CP sum, then DP average
    optimizer.step()
```

The optimizer is user-owned. A PP+TP stage may contain ordinary tensors and
DTensors; optimizers whose CUDA default chooses a foreach update should be
created with `foreach=False` unless their implementation separates those tensor
kinds into compatible parameter groups.

## Merge behavior

The DAG merge operation embeds safe text token ids, replaces modality
placeholder positions with projected encoder features, carries optional language
inputs forward, and produces `inputs_embeds`, `attention_mask`, and masked
`labels`. It is part of the graph so PP can place it at the correct boundary and
cross-mesh CP routing can transfer only locally owned feature rows.
