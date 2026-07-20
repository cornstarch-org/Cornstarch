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

Modules can be frozen independently with normal `requires_grad_(False)` or by
choosing which parameters enter the optimizer.

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
