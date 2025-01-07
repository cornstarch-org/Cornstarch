Using FSDP/DDP only may not be scalable depending on the infrastructure (e.g. slow inter-node networking).
Cornstarch provides 4D parallelism (data parallelism, tensor parallelism, pipeline parallelism, and context parallelism).

## Creating `MultimodalParallelPlugin`

Cornstarch allows per-modality parallelization specification using modular information in `MultimodalModel`.
[Recall](/using_cornstarch/creating_mllm) that a `MultimodalModel` is organized with multiple `ModalEncoderModule`s, one per modality encoder:

``` py
from cornstarch.models.multimodal_language_model import ModalEncoderModule, MultimodalModel

vision_encoder = ...
audio_encoder = ...
llm = ...

mllm = MultimodalModel(
    encoders={
        "vision": ModalEncoderModule(vision_encoder),
        "audio": ModalEncoderModule(audio_encoder),
    },
    language_model=llm,
)
```

Cornstarch provides the same architecture to specify parallelization per modality encoder and llm:

``` py hl_lines="8-15"
from cornstarch.plugin.multimodal_parallel_plugin import ModalParallelPlugin, MultimodalParallelPlugin
from colossalai.booster import Booster

vision_encoder_plugin = ModalParallelPlugin(...)
audio_encoder_plugin = ModalParallelPlugin(...)
language_model_plugin = ModalParallelPlugin(...)

mllm_plugin = MultimodalParallelPlugin(
    encoder_plugins={
        "vision": vision_encoder_plugin,
        "audio": audio_encoder_plugin,
    },
    language_model_plugin=language_model_plugin,
    ...
)

# Parallelize the model.
booster = Booster(plugin=mllm_plugin)
parallel_mllm, _* = booster.boost(model=mllm, ...)
```

!!! note

    All encoders defined when creating `MultimodalModel` should have its corresponding plugin, otherwise an exception will be raised during parallelization.

!!! note

    Parallelization is done lazily; the model is not parallelized until `colossalai.booster.Booster.boost()` is called.

!!! note

    Currently using `MultimodalParallelPlugin` forces to use pipeline parallelism, as encoders and the LLM should be pipelined in different stages.

The structure of `MultimodalParallelPlugin` exactly follows that of `MultimodalModel`.
Each encoder and the language model must have their own `ModalParallelPlugin`, which specifies how each modality should be parallelized.

## Specifying Parallelization

Each `ModalParallelPlugin` has four arguments for parallel configurations: `tp_size`, `sp_size`, `sequence_parallelism_mode`, and `pipeline_template`.
The arguments are mapped to the following three parallel dimensions:

- Tensor Parallelism (TP): `tp_size`
- Context Parallelism (CP): `sp_size` and `sequence_parallelism_mode` (use `sequence_parallelism` for backward compatibility)
- Pipeline Parallelism (PP): `pipeline_template`

### Tensor Parallelism

All embedding and linear layers are partitioned to tensor parallel ranks.
For attention layers, it is partitioned in head dimension; the number of heads of a model should be divisible to `tp_size`.

!!! note

    Currently specifying different number of `tp_size` to different encoders or LLM is not supported.

### Context Parallelism

Cornstarch supports Ulysses all-to-all style context parallelism and Llama context parallelism (head-by-head parallelism).
You can set `sequence_parallelism_mode` to `all_to_all` (Ulysses) or `ring_attn` (llama CP) to choose the context parallelism mechanism.

Encoders and the LLM can have different number of `sp_size`.

!!! note

    If `sp_size <= 1`, `sequence_parallelism_mode` is ignored.

!!! note

    Currently context parallelism is supported only for the LLM.

### Pipeline Parallelism

Cornstarch uses pipeline template to specify pipeline parallelism (adopted from [Oobleck](https://dl.acm.org/doi/abs/10.1145/3600006.3613152)), instead of simply having the number of pipeline stages, to let users to specify pipeline stages more freely.

A way of creating a pipeline template is as follows.

1. Get all layers required to be included in a template.
2. Split the layers into a list of sublayers properly, each of which will be a set of layers of a pipeline stage.
3. Create a `cornstarch.pipeline_template.PipelineTemplate` instance.

For HF models, Cornstarch provides a way of automatically getting all layers in a model:

``` py title="Getting layers from a HF model"
from cornstarch.pipeline_template import PipelineTemplate
from transformers.models.llama import LlamaForCausalLM

language_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
layers: list[str] = PipelineTemplate.get_modules(language_model)

# layers: ["model.embed_tokens",
#   "model.layers.0",
#   "model.layers.1",
#   ...
#   "model.layers.31",
#   "model.norm",
#   "lm_head"]
```

Split the list of layers to however you want as `list[list[str]]`.
For example, If you want to make a 2-stage pipeline template,

``` py
layers_per_stage = [
    layers[:17],
    layers[17:],
]
```

which will assign the `embed_tokens` layer and first 16 decoder layers to the first pipeline stage, all the others to the second pipeline stage.

Now create a pipeline template:

``` py
pipeline_template = PipelineTemplate(
    model_name=PipelineTemplate.get_model_name(language_model),
    modules_per_stage=layers_per_stage,
)
```

which will give you a 2-stage pipeline template for `LlamaForCausalLM`:
``` py
pipeline_template
PipelineTemplate(transformers.models.llama.modeling_llama.LlamaForCausalLM, 2 stages)
````

!!! note

    Cornstarch verifies if the pipeline template is for the given unimodal model by checking its name and modules_per_stage.
    It will raise an exception if a pipeline template for different model is given.

### Data Parallelism

Data Parallelism is not explictly specified by some arguments.
Instead, Cornstarch automatically infers how many data parallel replicas are needed by computing the number of ranks in a parallel multimodal LLM and divide the world size by it.

``` py title="An example of parallelization"
vision_encoder_plugin = ModalParallelPlugin(
    tp_size=2, sp_size=1,
    pipeline_template= # a pipeline template with 1 stage
)
audio_encoder_plugin = ModalParallelPlugin(
    tp_size=1, sp_size=1,
    pipeline_template= # a pipeline template with 1 stage
)
language_model_plugin = ModalParallelPlugin(
    tp_size=4, sp_size=2,
    pipeline_template= # a pipeline template with 3 stages
)

mllm_plugin = MultimodalParallelPlugin(
    encoder_plugins={
        "vision": vision_encoder_plugin,
        "audio": audio_encoder_plugin,
    },
    language_model_plugin=language_model_plugin,
    ...
)
```
The number of total ranks in the example above is 27 (2\*1\*1 + 1\*1\*1 + 4\*2\*3).
If 54 GPUs join the training, there will be 2 data parallel replicas.

!!! note

    Cornstarch does not optimize rank assignment and leaves it to user responsibility.
    The example above assigns 3 GPUs to the encoders and 8 GPUs to the LLM;
    if each node has 8 GPUs, cross-node GPUs may be assigned to the LLM (5 GPUs from one node, and 3 GPUs from another one).


## Running Parallelized Module

Pipeline parallelism interleaves forward passes and backward passes; therefore existing code for training (`loss = model(**inputs); loss.backward()`) is not compatible.
You have to use `Booster.execute_pipeline()` API to run the model:

``` py
outputs = booster.execute_pipeline(
    dataloader_iterator,
    model,
    crierion,
    optimizer,
    return_loss=True,
    return_outputs=False,
)

optimizer.step()
optimizer.zero_grad()
```

Refer to [Colossal-AI Booster API](https://colossalai.org/docs/basics/booster_api#usage) and examples for more details about the arguments.