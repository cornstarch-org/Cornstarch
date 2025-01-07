## Creating a Multimodal LLM

Cornstarch supports creating a modular multimodal LLM from HuggingFace models.
For example, you can create a vision-language model (VLM) with Llama 8b and ViT as follows:

``` py
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.vit.modeling_vit import ViTPreTrainedModel
from cornstarch.models.multimodal_language_model import ModalEncoderModule, MultimodalModel

llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
vision_encoder = ViTPreTrainedModel.from_pretrained("openai/clip-vit-large-patch14")

mllm = MultimodalModel(
    encoders={
        "vision": ModalEncoderModule(vision_encoder),
    },
    language_model=llm,
)
```

where `mllm` has `language_model` and `vision_encoder` modules.

## Model Architecture

A simplified `MultimodalModel` architecture is as follows:

```
cornstarch.MultimodalModel
├── vision_encoder (cornstarch.ModalEncoderModule)
│   ├── module (transformers.PreTrainedModel)
│   └── projector (cornstarch.MultimodalProjector)
├── audio_encoder (cornstarch.ModalEncoderModule)
│   ├── module (transformers.PreTrainedModel)
│   └── projector (cornstarch.MultimodalProjector)
├── whatever_encoder (cornstarch.ModalEncoderModule)
│   ├── module (transformers.PreTrainedModel)
│   └── projector (cornstarch.MultimodalProjector)
└── language_model (transformers.PreTrainedModel)
```

It has one `language_model` that represents the base LLM from HuggingFace `transformers`, and can have arbitrary number of encoders.
Encoders are stored as a dictionary in `MultimodalModel.encoders` where a key represents its encoder name (`vision`, `audio`, or `whatever` in the example above) and the corresponding value is a `ModalEncoderModule`.

`ModalEncoderModule` is a single modality encoder that includes an encoder and a projector.
An encoder module is from HuggingFace `transformers`, and Cornstarch provides the definition of the projector (`cornstarch.models.multimodal_language_model.modeling_multimodal_language_model.MultiomdalProjector`).

## Creating a Projector

Cornstarch provides two ways of creating `MultimodalProjector`.

``` py
class MultimodalProjector(PreTrainedModel):
    def __init__(self, config: MultimodalProjectorConfig, projection: Optional[nn.Module] = None): ...
```

First, you can simply wrap your own `torch.nn.Module` with `MultimodalProjector`.
When you provide your module to `projection` in creating a `MultimodalProjector` instance, Cornstarch will use the given module as a projector module.

The generated projector module should explicitly be given to `ModalEncoderModule`.

``` py
wrapped_projector_module = MultiodalProjector(your_config, my_projector_module)
encoder = ModalEncoderModule(module=my_encoder_module, projector=wrapped_projector_module)
```

Second, Cornstarch can automatically initialize a new projector if no projector is given in `ModalEncoderModule`:

``` py
encoder = ModalEncoderModule(module=my_encoder_module)
# which is equivalent to 
encoder = ModalEncoderModule(module=my_encoder_module, projector=None)
```

It adopts lazy initialization; a projector module is not initialized during creating a `ModalEncoderModule`.
Instead, when a `MultimodalModel` is created, it checks whether a projection module in the given `MultimodalProjector` is `None`, and creates a projector module if so.

`MultimodalModel` accepts two arguments for projector creation as you want: `init_projector_type` and `init_activation`:

``` py
class MultimodalModel(nn.Module):
    def __init__(
        self,
        ...,
        init_projector_type: str = "linear",
        init_activation: str = "gelu",
    ): ...
```

Currently `MultimodalModel` accepts either `linear` or `mlp` as an `init_projector_type`:

- `linear`: has a single `torch.nn.Linear` layer.
- `mlp`: has two `torch.nn.Linear` layers, where there is an activation layer as `init_activation` type in the middle.
The type of activations that Cornstarch supports is defined in [`transformers.activations.ACT2CLS`](https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/activations.py#L200).


## Callback Interface

Cornstarch provides callback interface to add multimodal specific features without modifying underlying unimodal modules.
For example, Llava-Next has to choose an image feature based on its select strategy, which its base vision encoder such as `CLIPVisionModel` does not have.
Callback is a great place to implement such features.

Cornstarch provides three types of callbacks in `ModalEncoderModule`:

``` py hl_lines="4 5 6"
ModalEncoderModule(
    model=vision_encoder,
    projector=vision_projector,
    preprocess_callback=preprocess_vision_callback,
    postprocess_module_callback=postprocess_vision_callback,
    postprocess_projector_callback=postprocess_projector_callback,
)
```

The execution order of callbacks and modules is as follows:

``` py
encoder: ModalEncoderModule
for encoder in mllm.encoders:
    preprocessed_encoder_input = encoder.preprocess_callback(encoder_input)
    encoder_output = encoder.module(preprocessed_encoder_input)
    postprocessed_encoder_output = encoder.postprocess_module_callback(encoder_output)
    module_output = encoder.projector(postprocessed_encoder_output)
    postprocessed_module_output = encoder.postprocess_projector_callback(module_output)

# merge a list of postprocessed_module_outputs to text_embedding
merged_input = merge(postprocessed_module_outputs, language_inputs_embedding)
output = language_model(merged_input)
```

### A Llava-Next example of utilizing callback interface

The original [`LlavaNextForConditionalGeneration.forward()`](https://github.com/huggingface/transformers/blob/5d7739f15a6e50de416977fe2cc9cb516d67edda/src/transformers/models/llava_next/modeling_llava_next.py#L346) is implemented as follows:

``` py hl_lines="13-26"
class LlavaNextForConditionalGeneration:
    def forward(
        self,
        ...
    ):
        ...

        if inputs_embeds is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None and pixel_values.size(0) > 0:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )

            # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
            image_features, feature_lens = self.pack_image_features(
                image_features,
                image_sizes,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_newline=self.image_newline,
            )

        # embed vision output result to inputs_embeds
        n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
        n_image_features = image_features.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        special_image_mask = (
            (input_ids == self.config.image_token_index)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(...)
        ...
```

The highlighted Llava-next specific feature can be implemented in a callback:

``` py hl_lines="13-21"
from typing import Optional
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

def postprocess_projector_callback(
    inputs: dict,
    output: BaseModelOutput | tuple,
    vision_feature_select_strategy: Optional[str] = None,
) -> BaseModelOutput | tuple:
    pixel_values = inputs.get("pixel_values", None)

    if pixel_values is not None and pixel_values.size(0) > 0:
        # output[0] == output.last_hidden_state
        image_features = output[0]

        # pack_image_features function should be borrowed from
        # the LlavaNextForConditionalGeneration class
        image_features, feature_lens = pack_image_features(
            image_features,
            image_sizes,
            vision_feature_select_strategy=vision_feature_select_strategy,
            image_newline=image_newline,
        )

    # replace output hidden state with postprocessed results
    if isinstance(output, ModelOutput):
        output.last_hidden_state = image_features
    else:
        output = (image_features,) + output[1:]

    return output
```
which can be used for any combination of a vision encoder and an LLM:

```py title="Llava-Next with CLIP+Mistral using Cornstarch"
clip = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
mistral = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

mllm = MultimodalModel(
    encoders={
        "vision": ModalEncoderModule(
            model=clip,
            postprocess_projector_callback=postprocess_projector_callback,
        )
    },
    language_model=mistral,
)
```

```py title="Llava-Next with Siglip+Llama using Cornstarch"
siglip = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
llama = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

mllm = MultimodalModel(
    encoders={
        "vision": ModalEncoderModule(
            model=siglip,
            postprocess_projector_callback=postprocess_projector_callback,
        )
    },
    language_model=llama,
)
```