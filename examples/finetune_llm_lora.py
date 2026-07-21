"""Fine-tune a lazily loaded language model with LoRA on one device.

The ordering in this example is intentional:

1. Construct only the Cornstarch base-model topology on ``meta``.
2. Record how the base checkpoint should be loaded.
3. Record the LoRA configuration while leaving adapter tensors unallocated.
4. Materialize the base checkpoint on the execution device. Cornstarch then
   injects randomly initialized LoRA tensors on that device.
5. Optionally overwrite those random tensors from a LoRA adapter checkpoint.

PEFT wraps adapted layers under ``base_layer``. Injecting it before Cornstarch
loads the base checkpoint would therefore change parameter paths too early and
break Hugging Face-to-Cornstarch checkpoint mapping. Deferring injection also
means distributed materialization can determine local layer ownership first,
although this example deliberately uses one device and ordinary PyTorch.

Run on one GPU::

    python -m examples.finetune_llm_lora --steps 3

The CPU option follows the same lifecycle and is useful as a small smoke test::

    python -m examples.finetune_llm_lora \
        --device cpu --dtype float32 --steps 1 --batch-size 1 --seq-len 8
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import tyro
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from cornstarch.models import (
    RepeatedLayerCompileConfig,
    configure_finetuning,
    from_hf_config,
)


DTypeName = Literal["float32", "bfloat16", "float16"]


def _torch_dtype(name: DTypeName) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _adapter_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        parameter for name, parameter in model.named_parameters() if "lora_" in name
    ]


def finetune(
    model_name_or_path: str = "hf-internal-testing/tiny-random-LlamaForCausalLM",
    device: str = "cuda:0",
    dtype: DTypeName = "bfloat16",
    adapter_checkpoint: Path | None = None,
    adapter_output: Path | None = None,
    steps: int = 3,
    batch_size: int = 2,
    seq_len: int = 32,
    learning_rate: float = 1e-4,
) -> None:
    """Load a base checkpoint lazily, then initialize or load LoRA adapters."""
    execution_device = torch.device(device)
    if execution_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        torch.cuda.set_device(execution_device)
    parameter_dtype = _torch_dtype(dtype)

    # Step 1: from_hf_config creates Cornstarch's topology on meta. No model
    # parameter storage is allocated on CPU or GPU at this point.
    root_config = AutoConfig.from_pretrained(model_name_or_path)
    config = getattr(root_config, "text_config", root_config)
    model = from_hf_config(
        config,
        model_kind="language",
        attn_implementation="eager",
        layer_compile_config=RepeatedLayerCompileConfig(enabled=False),
    )
    assert all(parameter.is_meta for parameter in model.parameters())

    # Step 2: record the base initialization source. The checkpoint is not read
    # until materialize(), so this still consumes no base-model parameter RAM.
    model.set_checkpoint_init(model_name_or_path=model_name_or_path)

    # Step 3: record LoRA intent while the base remains on meta. Cornstarch does
    # not inject PEFT yet, so there are deliberately no lora_A/lora_B tensors.
    configure_finetuning(
        model,
        "lora",
        lora_config=LoraConfig(
            target_modules="all-linear",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
        ),
    )
    assert not _adapter_parameters(model)
    assert all(parameter.is_meta for parameter in model.parameters())
    print("Base topology is on meta; LoRA configuration is queued without tensors.")

    # Step 4: materialize the base first. Its original parameter paths are used
    # for checkpoint loading; only afterward does the queued callback inject
    # randomly initialized LoRA parameters directly on the execution device.
    model.materialize(execution_device, dtype=parameter_dtype)
    adapter_parameters = _adapter_parameters(model)
    assert adapter_parameters
    assert all(not parameter.is_meta for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in adapter_parameters)
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" not in name
    )
    print(
        f"Base checkpoint materialized on {execution_device}; "
        f"created {sum(parameter.numel() for parameter in adapter_parameters):,} "
        "trainable LoRA parameters."
    )

    # Step 5 (optional): PEFT can only load adapter tensors after the adapter
    # modules exist. This overwrites their random initialization while leaving
    # the already-loaded base checkpoint untouched. The checkpoint must match
    # the LoRA target modules, rank, and alpha configured above.
    if adapter_checkpoint is not None:
        adapter_state = load_file(str(adapter_checkpoint), device="cpu")
        set_peft_model_state_dict(model, adapter_state, adapter_name="default")

    model.train()
    optimizer = torch.optim.AdamW(adapter_parameters, lr=learning_rate)
    for step in range(steps):
        input_ids = torch.randint(
            0,
            int(config.vocab_size),
            (batch_size, seq_len),
            device=execution_device,
        )
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"step={step + 1} loss={loss.item():.4f}")

    if adapter_output is not None:
        adapter_output.parent.mkdir(parents=True, exist_ok=True)
        adapter_state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in get_peft_model_state_dict(model).items()
        }
        save_file(adapter_state, str(adapter_output))
        print(f"Saved LoRA adapter tensors to {adapter_output}")


if __name__ == "__main__":
    tyro.cli(finetune)
