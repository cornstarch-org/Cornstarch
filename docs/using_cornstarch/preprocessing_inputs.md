# Preprocessing multimodal inputs

Cornstarch deliberately does not require a combined multimodal processor. Use
the tokenizer and modality processors appropriate for the selected component
models, then make their outputs agree on a placeholder contract.

## Placeholder contract

The tokenized text contains one placeholder position for every projected
modality feature row. `modality_token_ids` tells the merge node which token id
belongs to each modality.

```python
batch = {
    "input_ids": input_ids,       # includes image_token_id placeholders
    "labels": labels,
    "pixel_values": pixel_values,
}

merged = graph.merge_modality_encoder_outputs(
    language_model=language_model,
    input_ids=ExecutionFuture("input_ids"),
    labels=ExecutionFuture("labels"),
    modality_token_ids={"vision": image_token_id},
    encoder_outputs={"vision": vision_features},
)
```

The merge masks placeholders before token embedding, scatters projected features
into their positions, and masks those labels to `-100`. It rejects feature-count
or hidden-width mismatches instead of silently truncating data.

## Microbatches

A collator may return one batch dictionary or a list of batch dictionaries. The
list is one optimizer step's microbatch sequence. Cornstarch never guesses how
to split images, audio, or other modality tensors because only the application
knows their sample correspondence.

```python
def collate(samples):
    full_batch = application_collate(samples)
    return split_into_aligned_microbatches(full_batch)

loader = context.prepare_dataloader(
    dataset,
    batch_size=global_batch_per_replica,
    collate_fn=collate,
)
```

`prepare_dataloader()` always yields a list and applies DP sampling plus CP text
ownership independently to each microbatch.

## Context-parallel seam metadata

For a context-sharded modality projector, a collator can provide
`cp_modality_attention_masks`, mapping each modality name to its projected-token
mask. Cornstarch combines those offsets with the language model's text offsets
to route each projected row directly to the destination CP owner. If omitted,
the schedule can derive a left-packed one-row-per-placeholder mask; independently
padded or non-left-packed outputs should provide the explicit mask.
