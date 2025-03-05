!!! info

    [Cornstarch repository](https://github.com/cornstarch-org/Cornstarch) provides an end-to-end example
    in [`examples/pretrain_vlm.py`](https://github.com/cornstarch-org/Cornstarch/blob/main/examples/pretrain_vlm.py).
    

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

Cornstarch supports freezing a portion of the `MultimodalModel`.
For encoder (`ModalEncoderModule`), an encoder and a projector can individually be frozen:

``` py
mllm = MultimodalModule(
    encoders={
        "vision": ModalEncoderModule(...),
        "audio": ModalEncoderModule(...),
    },
    langauge_model=llm,
)

mllm.train(
    encoders_mode={
        "vision": (False, True), # encoder and projector, respectively
        "audio": (True, True),
    },
    llm_mode=False,
)
```

!!! info

    if `encoders_mode` is not given, the train mode of all encoders including projectors' is set to `llm_mode`.

    ```
    mllm.train(llm_mode=True)
    ```

    is equivalent to:

    ```
    mllm.train(
        {encoder: (True, True) for encoder in mllm.encoders},
        llm_mode=True,
    )
    ```

If the given encoder key does not exist in the `MultimodalModel`, it raises a `ValueError`.
For example, if you call `mllm.train("non_existing_encoder", mode=False)`, the encoder key `non_existing` does not exist in the `MultimodalModel` encoder dictionary, hence it raises an error.

!!! note

    PyTorch `torch.nn.Module.train(mode=False)` API cannot be used.
    It still computes gradients for frozen modules and you cannot get any benefits in computing time and memory consumption.

## Merging Encoder Outputs to LLM Embedding Space

When multimodal LLM forward is executed, modality encoders are executed first.
After that, LLM input embedding layer is executed, the modality encoder outputs is embedded into the LLM embedding space, and then execute the remaining LLM layers with modality encoder outputs and LLM embedding outputs.
Cornstarch follows HuggingFace's way of embedding: *injection mechanism* that injects modality encoder outputs to proper locations that user wants to put them to.
To specify where to put the modality encoder outputs, custom tokens must be added before running the model.
Cornstarch automatically adds the custom token information to the model when `MultimodalProcessor` is created, as described in [Preprocessing inputs page](../preprocessing_inputs).

To maintain the language model's original embedding table, Cornstarch exploits a tricky way of executing the input embedding.
If we execute the input embedding with `input_ids` where custom tokens are embedded, the embedding layer will raise an out of index exception.
To avoid such exception, Cornstarch first masks all custom tokens with 0 and executes the input embedding.
The result for the masked tokens will be replaced by the encoder outputs, thus we do not have to care about the result in the corresponding token indices.