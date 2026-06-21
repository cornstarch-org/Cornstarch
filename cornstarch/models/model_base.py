from __future__ import annotations

import copy
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import PretrainedConfig, PreTrainedModel

_logger = logging.getLogger(__name__)

from cornstarch.models.kernel_provider import get_hf_kernel
from cornstarch.models.lazy_init import InitializationPlan
from cornstarch.models.layer_compile import RepeatedLayerCompileConfig
from cornstarch.models.layer_offload import RepeatedLayerOffloadConfig
from cornstarch.models.state_mapping import StateDictPrefixMap


class CornstarchModelBase(nn.Module):
    """Common lifecycle and checkpoint surface for Cornstarch-owned models.

    Cornstarch models are built from Hugging Face configs and often reuse Hugging
    Face leaf modules, but the root module structure and execution loop are owned
    by Cornstarch. This base class holds the pieces that are shared across
    language, vision, and audio wrappers: the source HF config, the optional HF
    kernel id, the lazy materialization plan, and the prefix map that translates
    between Hugging Face checkpoint keys and Cornstarch's internal module names.

    Instances are expected to start on the ``meta`` device. Construction records
    module topology without allocating real storage; ``materialize()`` later
    turns those meta tensors into concrete tensors using one of three plans:
    empty allocation, default random initialization, or checkpoint assignment.
    This keeps large-model construction cheap and makes it possible to load HF
    state dicts without first creating a full Hugging Face root model.

    The public checkpoint API intentionally speaks Hugging Face formats.
    ``load_hf_state_dict()``, ``to_hf_state_dict()``, and ``save_pretrained()``
    translate key names at the boundary so callers can continue to use standard
    HF state dicts and serialization while Cornstarch keeps a module layout that
    exposes repeated layers for materialization, offload, and later parallelism.
    """

    supports_gradient_checkpointing = True

    def __init__(
        self,
        hf_config: PretrainedConfig,
        hf_to_cornstarch_prefixes: tuple[tuple[str, str], ...],
        hf_model_factory: Callable[[PretrainedConfig], PreTrainedModel],
        attn_implementation: str | None = None,
        init_plan: InitializationPlan | None = None,
        layer_offload_config: RepeatedLayerOffloadConfig | None = None,
        layer_compile_config: RepeatedLayerCompileConfig | None = None,
    ):
        """Attach config, lazy initialization policy, and HF key translation.

        Subclasses pass in the prefix mapping for their visible module layout and
        a lightweight factory that can recreate the matching Hugging Face model
        when serialization or deterministic buffer reconstruction needs it. The
        factory is not stored as a root model; it is only used at those lifecycle
        boundaries.
        """
        super().__init__()
        self.hf_config = hf_config
        self.config = hf_config
        self.attn_implementation = attn_implementation
        self.layer_offload_config = layer_offload_config
        self.layer_compile_config = layer_compile_config or RepeatedLayerCompileConfig()
        self.gradient_checkpointing = True
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

    @property
    def uses_layer_offload(self) -> bool:
        """Return whether repeated-layer CPU offload is enabled for forwards."""
        return self.layer_offload_config is not None and self.layer_offload_config.enabled

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Return whether activation checkpointing is enabled on this module."""
        return self.gradient_checkpointing

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable HF-compatible activation checkpointing for repeated layers."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Keep the HF API surface while checkpointing remains mandatory."""
        raise RuntimeError("Cornstarch repeated-layer forwards require activation checkpointing.")

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

    def materialize(
        self,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = None,
    ) -> CornstarchModelBase:
        """Materialize a meta model on the requested device using its init plan.

        When ``dtype`` is provided, parameters and buffers are allocated in that
        dtype directly, avoiding an extra ``.to(dtype)`` copy after
        materialization.  This is the hook the distributed ``apply_*`` path uses
        to materialize a sharded (DTensor) meta model in, e.g., bf16.
        """
        if not self._is_meta():
            return self

        device = torch.device(device)
        if self._init_plan.mode == "checkpoint":
            state_dict = self._load_checkpoint_state_dict(device, dtype)
            self.load_state_dict(state_dict, strict=True, assign=True)
            self._copy_deterministic_meta_buffers(device)
        elif self._init_plan.mode == "random":
            self._copy_deterministic_meta_buffers(device)
            self._materialize_empty(device, dtype)
            self._random_initialize()
            self._copy_constant_parameters(device, dtype)
        elif self._init_plan.mode == "empty":
            self._copy_deterministic_meta_buffers(device)
            self._materialize_empty(device, dtype)
        else:
            raise ValueError(f"Unknown initialization plan: {self._init_plan.mode}")

        return self

    def load_hf_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load or stage Hugging Face-format weights for this Cornstarch model."""
        mapped_state_dict = self._state_mapper.hf_to_cornstarch_state_dict(state_dict)
        if self._is_meta():
            if self._init_plan.mode != "empty":
                warnings.warn(
                    f"load_hf_state_dict() is replacing an existing "
                    f"{self._init_plan.mode!r} InitializationPlan with a new "
                    f"checkpoint plan. The previous plan will be discarded.",
                    stacklevel=2,
                )
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

    def train(self, mode: bool = True) -> CornstarchModelBase:
        """Set the training mode for Cornstarch-owned modules."""
        super().train(mode)
        return self

    @staticmethod
    def _offload_module_list_to_cpu(
        layers: nn.ModuleList, layer_indices: Iterable[int] | None = None
    ) -> None:
        """Move selected repeated layers to CPU without touching non-layer modules."""
        indices = range(len(layers)) if layer_indices is None else layer_indices
        for index in indices:
            layer = layers[index]
            if not any(param.is_meta for param in layer.parameters(recurse=True)):
                layer.to("cpu")

    @staticmethod
    def _materialize_module_list(layers: nn.ModuleList, device: torch.device) -> None:
        """Allocate or move repeated layers onto the requested device."""
        for layer in layers:
            tensors = list(layer.parameters(recurse=True)) + list(layer.buffers(recurse=True))
            if any(tensor.is_meta for tensor in tensors):
                layer.to_empty(device=device)
            else:
                layer.to(device)

    def _is_meta(self) -> bool:
        """Return whether every registered tensor still lives on the meta device."""
        tensors = list(self.parameters()) + list(self.buffers())
        return bool(tensors) and all(tensor.is_meta for tensor in tensors)

    def _to_hf_keys(self, keys: Iterable[str]) -> list[str]:
        """Translate internal key names into sorted Hugging Face key names."""
        return sorted(self._state_mapper.cornstarch_to_hf_key(key) for key in keys)

    def _is_offloaded_repeated_layer_tensor(self, name: str) -> bool:
        """Return whether a tensor belongs to a CPU-master repeated layer."""
        if not self.uses_layer_offload:
            return False
        return any(
            name == module_name or name.startswith(f"{module_name}.")
            for module_name in self._repeated_layer_module_names()
        )

    def _materialization_device_for_tensor(
        self, name: str, requested_device: torch.device
    ) -> torch.device:
        """Choose the allocation device for a tensor during materialization."""
        if self._is_offloaded_repeated_layer_tensor(name):
            assert self.layer_offload_config is not None
            return self.layer_offload_config.cpu_torch_device
        return requested_device

    def _repeated_layer_module_names(self) -> tuple[str, ...]:
        """Return root module names that contain independently scheduled layers."""
        return ()

    def _section_names(self) -> tuple[str, str, str]:
        """Return ``(pre_section, repeated_layers, post_section)`` attribute names.

        Subclasses declare their three-section module layout (a ``pre_*``
        ``ModuleDict``, a repeated-layer ``ModuleList``, and a ``post_*``
        ``ModuleDict``) so the distributed ``apply_*`` helpers can walk any
        Cornstarch model generically — one parallelization codepath for every
        HF family, no per-model policies.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _section_names() to support "
            f"distributed parallelism."
        )

    def _load_checkpoint_state_dict(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> Mapping[str, torch.Tensor]:
        """Load staged checkpoint tensors onto the materialization device."""
        if self._init_plan.state_dict is not None:
            return {
                key: tensor.to(
                    device=self._materialization_device_for_tensor(key, device),
                    dtype=dtype if dtype is not None and tensor.is_floating_point() else None,
                    non_blocking=True,
                )
                for key, tensor in self._init_plan.state_dict.items()
            }
        if self._init_plan.checkpoint_path is None:
            raise RuntimeError("Checkpoint initialization requires a state_dict or checkpoint_path.")
        checkpoint_device = "cpu" if self.uses_layer_offload else str(device)
        state_dict = self._state_mapper.hf_to_cornstarch_state_dict(
            load_file(str(self._init_plan.checkpoint_path), device=checkpoint_device)
        )
        return {
            key: tensor.to(
                device=self._materialization_device_for_tensor(key, device),
                dtype=dtype if dtype is not None and tensor.is_floating_point() else None,
                non_blocking=True,
            )
            for key, tensor in state_dict.items()
        }

    def _copy_deterministic_meta_buffers(self, device: torch.device) -> None:
        """Materialize deterministic helper buffers that are not in checkpoints."""
        if not any(buffer.is_meta for buffer in self.buffers()):
            return

        dtype = next(
            (tensor.dtype for tensor in self.parameters() if tensor.is_floating_point()),
            None,
        )
        hf_model = self._hf_model_factory(copy.deepcopy(self.hf_config))
        if dtype is None:
            hf_model.to(device=device)
        else:
            hf_model.to(device=device, dtype=dtype)

        for hf_prefix, cornstarch_prefix in self._state_mapper.pairs:
            hf_module_path = hf_prefix.rstrip(".")
            cornstarch_module_path = cornstarch_prefix.rstrip(".")
            if not hf_module_path or not cornstarch_module_path:
                continue
            try:
                hf_module = hf_model.get_submodule(hf_module_path)
            except AttributeError:
                warnings.warn(
                    f"HF model does not have a submodule at {hf_module_path!r} "
                    f"(mapped from Cornstarch prefix {cornstarch_module_path!r}). "
                    f"Deterministic buffers for this prefix will not be copied; "
                    f"those buffers will remain as zero tensors after materialization.",
                    stacklevel=2,
                )
                continue
            try:
                cornstarch_module = self.get_submodule(cornstarch_module_path)
            except AttributeError:
                warnings.warn(
                    f"Cornstarch model does not have a submodule at "
                    f"{cornstarch_module_path!r} (mapped from HF prefix "
                    f"{hf_module_path!r}). Deterministic buffers for this prefix "
                    f"will not be copied.",
                    stacklevel=2,
                )
                continue
            self._copy_meta_buffers(
                source_module=hf_module,
                target_module=cornstarch_module,
                target_path=cornstarch_module_path,
                device=device,
            )

    def _copy_meta_buffers(
        self,
        source_module: nn.Module,
        target_module: nn.Module,
        target_path: str,
        device: torch.device,
    ) -> None:
        """Copy deterministic non-checkpoint buffers from a concrete HF module."""
        for name, target_buffer in target_module.named_buffers(recurse=True):
            if not target_buffer.is_meta:
                continue
            source_buffer = source_module.get_buffer(name)
            full_name = f"{target_path}.{name}"
            replacement = source_buffer.to(
                device=self._materialization_device_for_tensor(full_name, device),
                dtype=target_buffer.dtype if target_buffer.is_floating_point() else None,
            )
            self._set_tensor(full_name, replacement, None)

    def _copy_constant_parameters(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> None:
        """Copy parameters for modules that lack ``reset_parameters``.

        Modules like ``RMSNorm`` initialize their weight to a constant (ones) in
        their constructor but expose no ``reset_parameters`` method, so
        ``_random_initialize`` leaves them at the uninitialized
        ``_materialize_empty`` value (garbage, which can be NaN). This builds a
        throwaway HF model — whose constant-init modules hold the correct values
        regardless of seed — and copies those values for any module under a
        mapped prefix that lacks ``reset_parameters``.
        """
        needs_copy = any(
            not hasattr(module, "reset_parameters")
            and any(p.numel() > 0 for p in module.parameters(recurse=False))
            for module in self.modules()
        )
        if not needs_copy:
            return

        target_dtype = dtype or next(
            (p.dtype for p in self.parameters() if p.is_floating_point()), None
        )
        hf_model = self._hf_model_factory(copy.deepcopy(self.hf_config))
        hf_model.to(device=device)

        for hf_prefix, cornstarch_prefix in self._state_mapper.pairs:
            hf_path = hf_prefix.rstrip(".")
            cs_path = cornstarch_prefix.rstrip(".")
            if not hf_path or not cs_path:
                continue
            try:
                hf_model.get_submodule(hf_path)  # existence check
                cs_module = self.get_submodule(cs_path)
            except AttributeError:
                continue

            for rel_name, child in cs_module.named_modules():
                if hasattr(child, "reset_parameters"):
                    continue
                for param_name, param in child.named_parameters(recurse=False):
                    full_cs = f"{cs_path}.{rel_name}.{param_name}" if rel_name else f"{cs_path}.{param_name}"
                    full_hf = f"{hf_path}.{rel_name}.{param_name}" if rel_name else f"{hf_path}.{param_name}"
                    try:
                        src = hf_model.get_parameter(full_hf)
                    except AttributeError:
                        continue
                    target_device = self._materialization_device_for_tensor(full_cs, device)
                    replacement = src.to(device=target_device, dtype=target_dtype or src.dtype)
                    self._set_tensor(full_cs, replacement, param.requires_grad)

        del hf_model

    def _materialize_empty(
        self, device: torch.device, dtype: torch.dtype | None = None
    ) -> None:
        """Replace meta parameters and buffers with empty tensors on a device.

        Submodules that own DTensor parameters (recorded by
        ``apply_tensor_parallel`` before materialization) are materialized with
        ``nn.Module.to_empty``, which preserves the DTensor sharding metadata and
        hooks — a plain ``torch.empty`` replacement would drop the DTensor
        wrapping and break tensor parallelism.  Remaining plain meta tensors are
        replaced directly.
        """
        # Materialize DTensor-owning submodules first via to_empty so their
        # sharding specs survive; this clears their meta flag.
        dtensor_modules: set[str] = set()
        for name, parameter in self.named_parameters(remove_duplicate=False):
            if hasattr(parameter, "_local_tensor"):
                dtensor_modules.add(name.rpartition(".")[0])
        for module_path in dtensor_modules:
            module = self.get_submodule(module_path) if module_path else self
            target_device = self._materialization_device_for_tensor(module_path, device)
            module.to_empty(device=target_device)
            if dtype is not None:
                module.to(dtype=dtype)

        for name, parameter in list(self.named_parameters(remove_duplicate=False)):
            if parameter.is_meta:
                target_device = self._materialization_device_for_tensor(name, device)
                target_dtype = dtype if dtype is not None and parameter.is_floating_point() else parameter.dtype
                self._set_tensor(name, torch.empty(parameter.shape, dtype=target_dtype, device=target_device), parameter.requires_grad)

        for name, buffer in list(self.named_buffers(remove_duplicate=False)):
            if buffer.is_meta:
                target_device = self._materialization_device_for_tensor(name, device)
                target_dtype = dtype if dtype is not None and buffer.is_floating_point() else buffer.dtype
                self._set_tensor(name, torch.empty(buffer.shape, dtype=target_dtype, device=target_device), None)

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
