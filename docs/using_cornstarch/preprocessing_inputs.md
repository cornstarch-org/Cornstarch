Running a multimodal model needs multimodal inputs.
Yet processors for each modality model do not have interaction required for multimodal model execution.

In this section, we introduce the basics of multimodal interaction and Cornstarch APIs for interaction.

## Interation between modality inputs

Multimodal LLMs merge modality encoder outputs into text embedidng and execute an LLM together.
In the text input, it is typical to use special tokens such as `<image>` to indicate this is where modality encoder outputs should be located:

``` py
from transformers.models.llava_next import LlavaNextProcessor

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(messages, return_tensors="pt")

# text:
# '<|start_header_id|>user<|end_header_id|>\n\n<image>\nDescribe this image.<|eot_id|>'
```

When this text is tokenized, the `<image>` token is replaced with multiple image token IDs as placeholders, where later modality encoder outputs are injected by replacing the hidden states of the corresponding location.

``` py
inputs = processor(images=image, text=text, return_tensors="pt")

# inputs["input_ids"]:
# tensor([128000, 128006,    882, 128007,    271, 128256, 128256, 128256, 128256,
#         128256, 128256, 128256, 128256, 128256, 128256, 128256, 128256, 128256,
#         128256, 128256, 128256, 128256, 128256, 128256, 128256, 128256, 128256,
#         ...])
# token 128256 (image special token) is repetitively added to the tokenized input_ids,
# where the number of special tokens is exactly the same with the number of image tokens
# that will be generated after executing the vision encoder.
```

## Cornstarch `MultimodalProcessor`

To support the same feature with multimodal-unaware processors, Cornstarch provides `MultimodalProcessor` class to define the multimodal interaction between processors, feature extractors, and a tokenizer.

``` py
from cornstarch.models.multimodal_language_model.processing_multimodal_language_model import MultimodalProcessor

mm_processor = MultimodalProcessor(
    encoder_processors={
        "vision": CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16"),
        "audio": WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3"),
    },
    llm_tokenizer=LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.1-8B-Instruct"),
    num_feature_calculation_funcs={},
)
```

which wraps a CLIP image processor for `vision` modality encoder, a Whisper feature extractor for `audio` modality encoder, and a Llama tokenizer for LLM into a single processor.

### Functions for calculating the number of features

In the example of using llava processor above, tokenized inputs has a lot of image tokens (128256).
The number of the image tokens must be exactly the same with the number of image features, otherwise merging the modality encoder outputs fails:

``` py
(inputs["input_ids"] == processor.tokenizer.convert_tokns_to_ids(processor.image_token)).sum()
# 1176 for image size 336x336

image_features.shape # during LlavaNextForConditionalGeneration.forward()
# torch.Size([1176, 4096])
```

Because Cornstarch does not know how many tokens the given model will generate, users need to provide functions that return the number of tokens in `num_features_calculation_funcs`:

``` py
MultimodalProcessor(
    ...
    num_feature_calculation_funcs: {
        "vision": lambda *args, **kwargs: (224 // 16) ** 2 + 1,
        "audio": lambda *args, **kwargs: 1500,
    }
)
```

Cornstarch provides processor inputs and processor outputs as two dictionaries as the input of the function:

``` py
inputs = # input to the modality encoder processor as a dictionary
outputs = # output of the modality encoder processor as a dictionary
num_features = callback(inputs, outputs)
```

where `num_features` should be either
- `list[int]`: a list of the number of features, one per modality input, across the entire batch, or
- `list[list[int]]`: a list of the lists of the number of features, one per modality input per batch.

!!! note
    For modality encoders that Cornstarch officially supports, calculation functions are automatically set.

    However, if your multimodal model needs more features (e.g. Llava-next's dynamic high resolution that its underlying CLIP vision encoder does not support) or if you use a modality encoder that Cornstarch does not know, a custom function must be provided.

### Token ID configuration

The tokenizer does not know which special tokens should be used for modality encoders.
At the same time, the LLM in `MultimodalModel` does not know which token IDs should be replaced with the modality encoder outputs when merging them, either.

For this reason, `MultimodalProcessor`, unlike processors that are independent from models in HuggingFace transformers, requires to take `MultimodalModel` to add such interaction:

``` py
class MultimodalProcessor:
    def __init__(
        self,
        ...
        model: MultimodalModel,
        num_features_calculation_funcs: dict[str, Callable] = {},
        predefined_tokens: dict[str, str] = {},
    ):
```

By default, Cornstarch registers `<modal_key>` special tokens to the tokenizer:

``` py hl_lines="5"
mm_processor.llm_tokenizer.special_tokens_map
{'bos_token': '<|begin_of_text|>',
 'eos_token': '<|eot_id|>',
 'unk_token': '<unk>',
 'additional_special_tokens': ['<vision>', '<audio>']} # Since we have "vision" and "audio" as modality keys, these two tokens are registered
```

When your dataset already includes its own special token, you can override the token by providing `predefined_tokens`.
The following example registers `<image>` instead of `<vision>` for the vision encoder:

``` py hl_lines="7"
mm_processor = MultimodalProcessor(..., predefined_tokens={"vision": "<image>"})

mm_processor.llm_tokenizer.special_tokens_map
{'bos_token': '<|begin_of_text|>',
 'eos_token': '<|eot_id|>',
 'unk_token': '<unk>',
 'additional_special_tokens': ['<image>', '<audio>']}
```

## Data preprocessing with `MultimodalProcessor`

Cornstarch designs the `MultimodalProcessor` to provide the maximum flexibility of data processing to users.
To avoid duplicated arguments from multiple modalities and the LLM, `MultimodalProcessor` takes a dictionary per modality encoder and the LLM:

``` py
class MultimodalProcessor:
    def __call__(
        self,
        encoder_inputs: dict[str, dict],
        llm_inputs: dict,
        return_tensors: str | TensorType = TensorType.PYTORCH,
        **kwargs,
    ) -> BatchFeature:
        ...
```

Cornstarch executes each processor and the tokenizer with the corresponding input dictionary.
It also forwards arguments in `kwargs` if a processor accepts the argument.
So, you do not have to repetitively include some common argument to dictionaries for multiple processors.