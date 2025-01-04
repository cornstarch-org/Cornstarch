## Training a Multimodal LLM

`MultimodalModel` inherits from `torch.nn.Module`, which has its own `forward()` function for inference and training.
You can call the model as you do with a typical `torch.nn.Module` as:

```py
mllm = MultimodalModel(...)
output = mllm(**inputs)
loss = output.loss
loss.backward()
# optimizer step, zero_grad, etc.
```

## Freezing Modules

Cornstarch supports freezing a part of the `MultimodalModel`.
For encoder (`ModalEncoderModule`), an encoder and a projector can individually be frozen:

``` py
mllm = MultimodalModule(
    encoders={
        "vision": ModalEncoderModule(...),
    },
    langauge_model=llm,
)

mllm.train("vision_encoder", mode=False)
mllm.train("vision_projector", mode=True)
mllm.train("language_model", mode=False)
```

Each modality encoder module (`ModalEncoderModule`) has two components: an encoder, and a projector.
The example above has a `vision` encoder module, where `vision_encoder` and `vision_projector` keys represent `mllm.vision_encoder.module` (the encoder) and `mllm.vision_encoder.projector` (the projector), respectively.

If the given encoder key does not exist in the `MultimodalModel`, it raises a `ValueError`.
For example, if you call `mllm.train("non_existing_encoder", mode=False)`, the encoder key `non_existing` does not exist in the `MultimodalModel` encoder dictionary, hence it raises an error.

!!! note

    PyTorch `torch.nn.Module.train(mode=False)` API cannot be used.
    It still computes gradients for frozen modules and you cannot get any benefits in computing time and memory consumption.

## Merging Encoder Outputs to LLM Embedding Space

When multimodal LLM forward is executed, modality encoders are executed first.
After that, LLM input embedding layer is executed, the modality encoder outputs is embedded into the LLM embedding space, and then execute the remaining LLM layers with modality encoder outputs and LLM embedding outputs.
Cornstarch provides two ways of embedding  modality encoder outputs to LLM embedding outputs: prepending and injecting.

### Prepending modality encoder outputs

Like initial VLMS (e.g. Llava 1.5), *prepending* simply attaches all modality encoder outputs prior to text embedding outputs.
The order of prepending is the opposite to the order of modality encoders.
When there are multiple modality encoders, encoders are executed in order of their order in the `MultimodalModule.encoders` dictionary.
If the output of the last modality encoder is prepended first, the result will be `[modality_encoder1][modality_encoder2]...[modality_encoderN][llm_embedding]`, which is the same as the order of modality encoders in the dictionary.

!!! note
    
    It does not require any specific LLM tokens to specify the location of modality encoders.

### Injecting modality encoder outputs

Unlike simply prepending modality encoders, *injecting` mehcanism injects modality encoder outputs to the location that user wants to put them to.
To specify where to put the modality encoder outputs, custom tokens must be added before running the model.
Use `MultimodalModel.set_token_ids()` API to register the token IDs per modality encoders:

``` py
mllm = MultimodalModel(
    encoders={
        "vision": ...,
        "audio": ...,
    },
    language_model=llm,
)

mllm.set_token_ids({
    "vision": vision_token_id,
    "audio": audio_token_id,
})
```

The model is automatically switched to injection mode and registered token IDs are used in merging modality encoder outputs.

It is a user responsibility to prepare placeholders that modality encoder outputs will replace.
During injection, Cornstarch replaces embeddings in the indices of the corresponding token IDs with the modality encoder outputs. For example, if the vision token ID is 200 and the given `input_ids` (tokenized text) looks like:

```
input_ids = [1, 33, 7752, 10452, 200, 200, 200, 200, 4465, 4832, 6921, ...]
```
You are expecting there would be 4 vision tokens from the vision encoder, which will be injected at indices 4, 5, 6, and 7 (zero-based).

!!! note

    The number of tokens must match.