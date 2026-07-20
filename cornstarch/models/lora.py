"""LoRA adapter attachment that preserves Cornstarch model interfaces."""

from __future__ import annotations

import copy

import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

from cornstarch.models.model_base import CornstarchModelBase
from cornstarch.models.multimodal.modeling import CornstarchModalityEncoder


def attach_lora(
    module: nn.Module,
    config: LoraConfig,
    *,
    adapter_name: str = "default",
) -> nn.Module:
    """Attach a PEFT LoRA adapter in place and return ``module`` unchanged.

    Passing a :class:`CornstarchModalityEncoder` targets its encoder only; its
    projector remains an ordinary independently trainable module.  Passing a
    language model targets that model directly, so encoder and language-model
    adapters can be selected independently.

    Cornstarch models begin on the ``meta`` device.  For those models adapter
    injection is deferred until base checkpoint or random initialization has
    completed.  This preserves Hugging Face checkpoint key translation while
    keeping the call site identical for local and distributed materialization.
    PEFT injects into already-materialized modules immediately.
    """
    if not isinstance(config, LoraConfig):
        raise TypeError(
            f"config must be a peft.LoraConfig, got {type(config).__name__}."
        )
    if not adapter_name:
        raise ValueError("adapter_name must not be empty.")

    target = module.encoder if isinstance(module, CornstarchModalityEncoder) else module
    adapter_names: set[str] = getattr(target, "_cornstarch_lora_adapter_names", set())
    if adapter_name in adapter_names:
        raise ValueError(
            f"LoRA adapter {adapter_name!r} is already attached to "
            f"{type(target).__name__}."
        )

    deferred_config = copy.deepcopy(config)

    def inject(target_module: nn.Module) -> None:
        inject_adapter_in_model(
            deferred_config,
            target_module,
            adapter_name=adapter_name,
        )

    if isinstance(target, CornstarchModelBase):
        target._register_post_materialize_callback(inject)
    elif _has_meta_tensors(target):
        raise RuntimeError(
            "Cannot attach LoRA to a generic module on the meta device. "
            "Materialize it first, or pass a Cornstarch model so injection can "
            "be deferred safely."
        )
    else:
        inject(target)

    adapter_names.add(adapter_name)
    target._cornstarch_lora_adapter_names = adapter_names
    return module


def _has_meta_tensors(module: nn.Module) -> bool:
    return any(tensor.is_meta for tensor in module.parameters()) or any(
        tensor.is_meta for tensor in module.buffers()
    )
