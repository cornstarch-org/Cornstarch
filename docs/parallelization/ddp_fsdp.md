!!! info

    [Cornstarch repository](https://github.com/cornstarch-org/Cornstarch) provides an end-to-end example
    in [`examples/distributed/run_vlm_ddp.py`](https://github.com/cornstarch-org/Cornstarch/blob/main/examples/distributed/run_vlm_ddp.py) and [`examples/distributed/run_vlm_fsdp.py`](https://github.com/cornstarch-org/Cornstarch/blob/main/examples/distributed/run_vlm_fsdp.py).

PyTorch DDP and FSDP work by simply wrapping the original model with the API.
This design principle is also compatible with Cornstarch multimodal LLM, therefore DDP/FSDP can be used with Cornstarch.

``` py title="An example of using PyTorch DDP" hl_lines="43 46"
import torch
import torch.distributed as dist
from torch.optim.adam import Adam
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
)
from transformers.models.clip import CLIPVisionModel, CLIPImageProcessor

from cornstarch.models.multimodal_language_model import (
    ModalEncoderModule,
    MultimodalModel,
    MultimodalModelProcessor,
)

# Create a mllm
vision_model_name = "openai/clip-vit-base-patch32"
language_model_name = "meta-llama/Llama-3.2-1B"
vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
language_model = AutoModelForCausalLM.from_pretrained(language_model_name)

model = MultimodalModel(
    encoders={"vision": vision_encoder},
    language_model=language_model,
).to(dtype=torch.bfloat16, device="cuda")

# Create a processor
image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
text_tokenizer = AutoTokenizer.from_pretrained(language_model_name, use_fase=True)
text_processor.pad_token_id = text_processor.eos_token_id

processor = MultimodalModelProcessor(
    tokenizer=text_tokenizer,
    image_processor=image_processor,
)

# Parallelize the model
dist.init_process_group()
ddp_model = DistributedDataParallel(model)
optimizer = Adam(ddp_model.parameters())

outputs = ddp_model(**inputs)
loss = outputs.loss
loss.backward()

optimizer.step()
optimizer.zero_grad()
```

Similarly, FSDP can be used by wrapping the model with `torch.distributed._composable.fsdp.fully_shard()`:

!!! note

    It is very important to properly define a wrapping unit using `ModuleWrapPolicy` (FSDP1) or `fully_shard` (FSDP2) in performance and correctness.
    Parameters with different `requires_grad` cannot be wrapped together, thus they need to be wrapped in a different group; otherwise it will raise an error.
    Because of this, using FSDP still requires knowledge of model internal architecture.

``` py title="An example of using PyTorch FSDP1"
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
# All the same before dist.init_process_group()

dist.init_process_group()

fsdp_model = FullyShardedDataParallel(
    module=model,                           # required
    auto_wrap_policy=ModuleWrapPolicy(      # required
        [
            ModalEncoderModule,
            MultimodalProjector,
            torch.nn.Embedding,
            CLIPEncoderLayer,               
            LlamaDecoderLayer,
        ]
    ),
    sharding_strategy=ShardingStrateegy.FULL_SHARD,  # optional
    cpu_offload=CPUOffload(),                        # optional
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # optional
    forward_prefetch=True,                           # optional
)
optimizer = Adam(fsdp_model.parameters())

outputs = fsdp_model(**inputs)
loss = outputs.loss
loss.backward()

optimizer.step()
optimizer.zero_grad()
```

``` py title="An example of using PyTorch FSDP2"
from torch.distributed._composable.fsdp import fully_shard

vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
language_model = AutoModelForCausalLM.from_pretrained(language_model_name)

# The location of fully_shard() for subgroups does not have to be here.
for layer in vision_encoder.vision_model.encoder.layers:
    fully_shard(layer)
fully_shard(vision_encoder.vision_model)
for layer in language_model.model.layers:
    fully_shard(layer)

model = MultimodalModel(
    encoders={"vision": vision_encoder},
    language_model=language_model,
).to(dtype=torch.bfloat16, device="cuda")
fully_shard(model.vision_encoder.projector)
fsdp_model = fully_shard(model)

dist.init_process_group()

optimizer = Adam(fsdp_model.parameters())

outputs = fsdp_model(**inputs)
loss = outputs.loss
loss.backward()

optimizer.step()
optimizer.zero_grad()
```