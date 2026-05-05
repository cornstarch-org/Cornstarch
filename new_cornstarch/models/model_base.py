from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel

from new_cornstarch.models.kernel_provider import get_hf_kernel
from new_cornstarch.models.lazy_init import InitializationPlan
from new_cornstarch.models.state_mapping import StateDictPrefixMap


class CornstarchModelBase(nn.Module):
    """Base class for Cornstarch-owned models with HF checkpoint mapping."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        attn_implementation: str | None = None,
        init_plan: InitializationPlan | None = None,
    ):
        """Attach shared config, attention kernel, init plan, and state mapping."""
        super().__init__()
        self.hf_config = hf_config
        self.config = hf_config
        self.attn_implementation = attn_implementation
        self._attention_kernel = None
        self._init_plan = init_plan or InitializationPlan.empty()
        self._state_mapper = StateDictPrefixMap(hf_to_cornstarch_prefixes)
        self._hf_model_factory = hf_model_factory

    @property
    def attention_kernel(self) -> Any:
        """Lazily load the configured Hugging Face kernel implementation."""
        if self.attn_implementation is None:
            raise ValueError("No attention kernel was configured for this Cornstarch model.")
        if self._attention_kernel is None:
            self._attention_kernel = get_hf_kernel(self.attn_implementation)
        return self._attention_kernel

    def set_empty_init(self) -> None:
        """Configure materialization to allocate tensors without initializing them."""
        self._init_plan = InitializationPlan.empty()

    def set_random_init(self) -> None:
        """Configure materialization to initialize tensors with model defaults."""
        self._init_plan = InitializationPlan.random()

    def set_checkpoint_init(
        self,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Configure materialization to assign weights from a state dict or file."""
        if state_dict is not None:
            state_dict = self._state_mapper.hf_to_cornstarch_state_dict(state_dict)
        self._init_plan = InitializationPlan.checkpoint(state_dict=state_dict, checkpoint_path=checkpoint_path)

    def materialize(self, device: str | torch.device = "cuda") -> CornstarchModelBase:
        """Materialize a meta model on the requested device using its init plan."""
        if not self._is_meta():
            return self

        device = torch.device(device)
        if self._init_plan.mode == "checkpoint":
            state_dict = self._load_checkpoint_state_dict(device)
            self.load_state_dict(state_dict, strict=True, assign=True)
        elif self._init_plan.mode == "random":
            self._materialize_empty(device)
            self._random_initialize()
        elif self._init_plan.mode == "empty":
            self._materialize_empty(device)
        else:
            raise ValueError(f"Unknown initialization plan: {self._init_plan.mode}")

        return self

    def load_hf_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load or stage Hugging Face-format weights for this Cornstarch model."""
        mapped_state_dict = self._state_mapper.hf_to_cornstarch_state_dict(state_dict)
        if self._is_meta():
            self._init_plan = InitializationPlan.checkpoint(state_dict=mapped_state_dict)
            expected = set(self.state_dict().keys())
            actual = set(mapped_state_dict.keys())
            missing = self._to_hf_keys(expected - actual)
            unexpected = self._to_hf_keys(actual - expected)
            if strict and (missing or unexpected):
                raise RuntimeError(f"State dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
            return missing, unexpected

        incompatible = self.load_state_dict(mapped_state_dict, strict=strict)
        return self._to_hf_keys(incompatible.missing_keys), self._to_hf_keys(incompatible.unexpected_keys)

    def to_hf_state_dict(self) -> dict[str, torch.Tensor]:
        """Return this model's tensors in Hugging Face state-dict key space."""
        if self._is_meta():
            raise RuntimeError("Cannot export an HF state dict before materialize().")
        return self._state_mapper.cornstarch_to_hf_state_dict(self.state_dict())

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save the model using Hugging Face serialization and key layout."""
        if self._is_meta():
            raise RuntimeError("Cannot save a meta model. Call materialize() first.")

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        hf_model = self._hf_model_factory(copy.deepcopy(self.hf_config))
        hf_model.load_state_dict(
            {key: tensor.detach().cpu() for key, tensor in self.to_hf_state_dict().items()},
            strict=True,
        )
        generation_config = getattr(hf_model, "generation_config", None)
        original_pad_token_id = None
        if generation_config is not None:
            original_pad_token_id = generation_config.pad_token_id
            if generation_config.pad_token_id is not None and generation_config.pad_token_id < 0:
                generation_config.pad_token_id = generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = generation_config.bos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = 0

        try:
            hf_model.save_pretrained(save_directory, **kwargs)
        finally:
            if generation_config is not None:
                generation_config.pad_token_id = original_pad_token_id

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run model-specific forward logic implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} does not implement forward().")

    def _is_meta(self) -> bool:
        """Return whether every registered tensor still lives on the meta device."""
        tensors = list(self.parameters()) + list(self.buffers())
        return bool(tensors) and all(tensor.is_meta for tensor in tensors)

    def _to_hf_keys(self, keys: Iterable[str]) -> list[str]:
        """Translate internal key names into sorted Hugging Face key names."""
        return sorted(self._state_mapper.cornstarch_to_hf_key(key) for key in keys)

    def _load_checkpoint_state_dict(self, device: torch.device) -> Mapping[str, torch.Tensor]:
        """Load staged checkpoint tensors onto the materialization device."""
        if self._init_plan.state_dict is not None:
            return {
                key: tensor.to(device=device, non_blocking=True)
                for key, tensor in self._init_plan.state_dict.items()
            }
        if self._init_plan.checkpoint_path is None:
            raise RuntimeError("Checkpoint initialization requires a state_dict or checkpoint_path.")
        state_dict = load_file(str(self._init_plan.checkpoint_path), device=str(device))
        return self._state_mapper.hf_to_cornstarch_state_dict(state_dict)

    def _materialize_empty(self, device: torch.device) -> None:
        """Replace meta parameters and buffers with empty tensors on a device."""
        for name, parameter in list(self.named_parameters()):
            if parameter.is_meta:
                self._set_tensor(name, torch.empty(parameter.shape, dtype=parameter.dtype, device=device), parameter.requires_grad)

        for name, buffer in list(self.named_buffers()):
            if buffer.is_meta:
                self._set_tensor(name, torch.empty(buffer.shape, dtype=buffer.dtype, device=device), None)

    def _random_initialize(self) -> None:
        """Run the best available PyTorch initialization hooks."""
        for module in self.modules():
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

    def _set_tensor(
        self, name: str, tensor: torch.Tensor, requires_grad: bool | None
    ) -> None:
        """Install a parameter or buffer tensor at a dotted module path."""
        module_name, _, tensor_name = name.rpartition(".")
        module: nn.Module = self.get_submodule(module_name) if module_name else self
        if requires_grad is None:
            module._buffers[tensor_name] = tensor
        else:
            module._parameters[tensor_name] = nn.Parameter(tensor, requires_grad=requires_grad)
