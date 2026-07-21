from __future__ import annotations

import copy
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import PretrainedConfig, PreTrainedModel

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
        self._post_materialize_callbacks: list[
            Callable[[CornstarchModelBase], None]
        ] = []

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
        model_name_or_path: str | None = None,
    ) -> None:
        """Configure materialization to assign weights from checkpoint sources.

        Exactly one source is used: a pre-loaded Hugging Face ``state_dict``, a
        local safetensors ``checkpoint_path``, or a Hugging Face Hub
        ``model_name_or_path``. Safetensors are opened lazily at
        ``materialize()`` time: a PP stage reads only its global layer keys, and
        TP moves only this rank's tensor slices to the target device.
        """
        if state_dict is not None:
            state_dict = self._state_mapper.hf_to_cornstarch_state_dict(state_dict)
        self._init_plan = InitializationPlan.checkpoint(
            state_dict=state_dict,
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_name_or_path,
        )

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
            self._run_post_materialize_callbacks()
            return self

        device = torch.device(device)
        if self._init_plan.mode == "checkpoint":
            state_dict = self._load_checkpoint_state_dict()
            self._load_checkpoint_into_model(state_dict, device, dtype)
            self._copy_deterministic_meta_buffers(device)
        elif self._init_plan.mode == "random":
            self._copy_deterministic_meta_buffers(device)
            self._materialize_empty(device, dtype)
            self._random_initialize()
            self._copy_local_tp_parameters(device, dtype)
            self._copy_constant_parameters(device, dtype)
        elif self._init_plan.mode == "empty":
            self._copy_deterministic_meta_buffers(device)
            self._materialize_empty(device, dtype)
        else:
            raise ValueError(f"Unknown initialization plan: {self._init_plan.mode}")

        self._run_post_materialize_callbacks()
        return self

    def _register_post_materialize_callback(
        self, callback: Callable[[CornstarchModelBase], None]
    ) -> None:
        """Run ``callback`` after this model's base tensors materialize.

        Structural extensions such as PEFT adapters must not wrap the lazy model
        before checkpoint keys have been translated and loaded.  This private
        lifecycle hook lets those extensions declare their intent while the
        model is still on ``meta`` and apply the mutation at the safe boundary.
        """
        if self._is_meta():
            self._post_materialize_callbacks.append(callback)
        else:
            callback(self)

    def _run_post_materialize_callbacks(self) -> None:
        """Apply and clear structural mutations queued for materialization."""
        callbacks = self._post_materialize_callbacks
        if not callbacks:
            return
        for callback in callbacks:
            callback(self)
        self._post_materialize_callbacks = []

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

    def offload_layers_to_cpu(
        self, layer_indices: Iterable[int] | None = None
    ) -> None:
        """Move selected repeated layers to CPU without family-specific code.

        Every unified Cornstarch transformer exposes exactly one repeated stack
        through ``_section_names``. Memory policy operates on that structural
        contract; it never needs to know whether the source checkpoint was
        Llama, CLIP, Whisper, or another Hugging Face family.
        """
        layers = self._repeated_layers()
        indices = range(len(layers)) if layer_indices is None else layer_indices
        for index in indices:
            layer = layers[index]
            if not any(param.is_meta for param in layer.parameters(recurse=True)):
                layer.to("cpu")

    def materialize_layers(self, device: str | torch.device) -> None:
        """Allocate or move the unified repeated stack onto ``device``."""
        if self.uses_layer_offload:
            assert self.layer_offload_config is not None
            device = self.layer_offload_config.cpu_torch_device
        device = torch.device(device)
        for layer in self._repeated_layers():
            tensors = list(layer.parameters(recurse=True)) + list(
                layer.buffers(recurse=True)
            )
            if any(tensor.is_meta for tensor in tensors):
                layer.to_empty(device=device)
            else:
                layer.to(device)

    def _repeated_layers(self) -> nn.ModuleList:
        """Return the repeated section shared by lifecycle and parallelism."""
        layers = getattr(self, self._section_names()[1])
        if not isinstance(layers, nn.ModuleList):
            raise TypeError("A Cornstarch repeated section must be an nn.ModuleList.")
        return layers

    def _is_meta(self) -> bool:
        """Return whether any registered tensor still lives on the meta device."""
        tensors = list(self.parameters()) + list(self.buffers())
        return any(tensor.is_meta for tensor in tensors)

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
        """Return the unified repeated-section name for device placement."""
        return (self._section_names()[1],)

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

    def _load_checkpoint_state_dict(self) -> Mapping[str, torch.Tensor]:
        """Read only tensors owned by this PP stage, keyed by local model name.

        Checkpoint files use global Hugging Face layer indices, while PP slices
        the repeated ``ModuleList`` and renumbers it from zero.  Build the exact
        local-to-global key map from the already-parallelized meta model before
        touching storage.  TP slicing happens in ``_load_checkpoint_into_model``
        so a full source tensor may exist briefly on CPU, but never on every
        rank's accelerator.

        Device and dtype placement is intentionally deferred until rank
        ownership is known.
        """
        local_to_global = {
            local_key: self._global_cornstarch_key(local_key)
            for local_key in self.state_dict()
        }
        if self._init_plan.state_dict is not None:
            source = self._init_plan.state_dict
            return self._select_checkpoint_tensors(source, local_to_global)

        if self._init_plan.model_name_or_path is not None:
            hf_to_local = {
                self._state_mapper.cornstarch_to_hf_key(global_key): local_key
                for local_key, global_key in local_to_global.items()
            }
            return self._download_and_load_safetensors(
                self._init_plan.model_name_or_path, hf_to_local
            )
        if self._init_plan.checkpoint_path is not None:
            hf_to_local = {
                self._state_mapper.cornstarch_to_hf_key(global_key): local_key
                for local_key, global_key in local_to_global.items()
            }
            return self._read_safetensors(
                str(self._init_plan.checkpoint_path), hf_to_local
            )
        raise RuntimeError(
            "Checkpoint initialization requires a state_dict, checkpoint_path, "
            "or model_name_or_path."
        )

    @staticmethod
    def _select_checkpoint_tensors(
        source: Mapping[str, torch.Tensor],
        local_to_source: Mapping[str, str],
    ) -> dict[str, torch.Tensor]:
        """Select a rank's local keys from an already-loaded checkpoint."""
        missing = [
            source_key
            for source_key in local_to_source.values()
            if source_key not in source
        ]
        if missing:
            raise RuntimeError(f"Checkpoint is missing tensors: {missing[:5]}")
        return {
            local_key: source[source_key]
            for local_key, source_key in local_to_source.items()
        }

    @staticmethod
    def _read_safetensors(
        path: str, source_to_local: Mapping[str, str]
    ) -> dict[str, torch.Tensor]:
        """Read requested tensors from one safetensors file onto CPU."""
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            for source_key, local_key in source_to_local.items():
                if source_key in available:
                    tensors[local_key] = checkpoint.get_tensor(source_key)
        return tensors

    @staticmethod
    def _download_and_load_safetensors(
        model_name_or_path: str,
        source_to_local: Mapping[str, str],
    ) -> dict[str, torch.Tensor]:
        """Download HF shards as needed and read only locally-owned tensors.

        Files may still enter the shared Hub cache, but tensors not owned by this
        PP stage are never materialized.  Reading on CPU also lets TP take its
        local slice before transferring anything to the accelerator.
        """
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        siblings = api.model_info(model_name_or_path).siblings or []
        safetensor_files = sorted(
            sibling.rfilename
            for sibling in siblings
            if sibling.rfilename.endswith(".safetensors")
        )
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in '{model_name_or_path}'."
            )

        selected: dict[str, torch.Tensor] = {}
        for filename in safetensor_files:
            local_path = hf_hub_download(model_name_or_path, filename)
            remaining = {
                source_key: local_key
                for source_key, local_key in source_to_local.items()
                if local_key not in selected
            }
            selected.update(CornstarchModelBase._read_safetensors(local_path, remaining))
            if len(selected) == len(source_to_local):
                break
        missing = sorted(set(source_to_local.values()) - set(selected))
        if missing:
            raise RuntimeError(f"Checkpoint is missing tensors: {missing[:5]}")
        return selected

    def _load_checkpoint_into_model(
        self,
        state_dict: Mapping[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> None:
        """Assign checkpoint tensors into this model, sharding DTensor params.

        For a plain (non-parallelized) model this is a strict ``assign``-load.
        For a tensor-parallel model, ``apply_tensor_parallel`` has recorded
        DTensor sharding specs on the meta parameters; a strict ``assign``-load of
        a *full* tensor would drop that wrapping and leave every rank with the
        whole weight. Instead each DTensor-owning module is materialized first via
        ``to_empty`` (which preserves the sharding spec), then this rank's slice
        is derived on CPU and only that slice is copied to the target device. The
        remaining plain params (and any persistent
        buffers carried in the checkpoint) are ``assign``-loaded; non-persistent
        deterministic buffers are intentionally left meta so the caller's
        ``_copy_deterministic_meta_buffers`` can fill them just as on the plain
        path.
        """
        from torch.distributed.tensor import DTensor

        params = dict(self.named_parameters(remove_duplicate=False))
        local_specs: dict[str, tuple] = getattr(
            self, "_local_tp_shard_specs", {}
        )
        dtensor_keys = [
            name for name, param in params.items()
            if isinstance(param.data, DTensor)
        ]
        local_keys = [name for name in params if name in local_specs]
        if not dtensor_keys and not local_keys:
            local_state = {
                key: self._move_checkpoint_tensor(key, tensor, device, dtype)
                for key, tensor in state_dict.items()
            }
            self.load_state_dict(local_state, strict=True, assign=True)
            return

        dtensor_key_set = set(dtensor_keys)
        with torch.no_grad():
            # ``to_empty`` preserves the DTensor wrapper. Deduplicate module
            # paths before copying: calling it twice on one module would erase
            # a parameter copied by an earlier iteration.
            module_paths = dict.fromkeys(key.rpartition(".")[0] for key in dtensor_keys)
            for module_path in module_paths:
                module = self.get_submodule(module_path) if module_path else self
                module.to_empty(
                    device=self._materialization_device_for_tensor(module_path, device)
                )
                if dtype is not None:
                    module.to(dtype=dtype)
            for key in dtensor_keys:
                target = params[key]
                local = self._local_dtensor_slice(
                    state_dict[key], target.data.device_mesh, target.data.placements
                )
                local = self._move_checkpoint_tensor(key, local, device, dtype)
                target.data.to_local().copy_(local)
            for key in local_keys:
                full = state_dict[key]
                local = self._slice_local_tp_tensor(full, local_specs[key])
                target = params[key]
                local = self._move_checkpoint_tensor(key, local, device, dtype)
                self._set_tensor(key, local, target.requires_grad)

        # Assign remaining plain parameters and persistent buffers. DTensor and
        # manual fused-section keys were installed above; deterministic,
        # non-persistent buffers remain meta for reconstruction by the caller.
        distributed_key_set = dtensor_key_set | set(local_keys)
        remaining = {
            key: self._move_checkpoint_tensor(key, value, device, dtype)
            for key, value in state_dict.items()
            if key not in distributed_key_set
        }
        self.load_state_dict(remaining, strict=False, assign=True)

    def _move_checkpoint_tensor(
        self,
        key: str,
        tensor: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        """Move one already-selected local tensor to its owning device."""
        return tensor.to(
            device=self._materialization_device_for_tensor(key, device),
            dtype=dtype if dtype is not None and tensor.is_floating_point() else None,
            non_blocking=True,
        )

    @staticmethod
    def _local_dtensor_slice(tensor: torch.Tensor, mesh, placements) -> torch.Tensor:
        """Derive a DTensor rank's local shard without a full-device collective."""
        from torch.distributed.tensor.placement_types import Replicate, Shard

        local = tensor
        for mesh_dim, placement in enumerate(placements):
            if isinstance(placement, Replicate):
                continue
            if not isinstance(placement, Shard):
                raise NotImplementedError(
                    f"Checkpoint loading does not support {placement!r} placement."
                )
            rank = mesh.get_local_rank(mesh_dim)
            local = local.chunk(mesh.size(mesh_dim), dim=placement.dim)[rank]
        return local.contiguous()

    @staticmethod
    def _slice_local_tp_tensor(tensor: torch.Tensor, spec: tuple) -> torch.Tensor:
        """Apply a recorded non-DTensor TP shard spec to a full HF tensor."""
        kind, dim, *payload = spec
        if kind == "chunk":
            rank, size = payload
            return tensor.chunk(size, dim=dim)[rank].contiguous()
        if kind == "sections":
            section_sizes, rank, size = payload
            sections = tensor.split(tuple(section_sizes), dim=dim)
            return torch.cat(
                [section.chunk(size, dim=dim)[rank] for section in sections],
                dim=dim,
            ).contiguous()
        raise ValueError(f"Unknown local TP shard spec {spec!r}.")

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
            source_name = self._global_repeated_layer_name(target_path, name)
            source_buffer = source_module.get_buffer(source_name)
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
            cs_module = self._constant_parameter_module(hf_model, hf_path, cs_path)
            if cs_module is None:
                continue
            self._copy_constant_module_parameters(
                hf_model, cs_module, hf_path, cs_path, device, target_dtype
            )

        del hf_model

    def _constant_parameter_module(
        self, hf_model: nn.Module, hf_path: str, cs_path: str
    ) -> nn.Module | None:
        """Resolve one mapped module pair, ignoring absent compatibility paths."""
        if not hf_path or not cs_path:
            return None
        try:
            hf_model.get_submodule(hf_path)
            return self.get_submodule(cs_path)
        except AttributeError:
            return None

    def _copy_constant_module_parameters(
        self,
        hf_model: nn.Module,
        cs_module: nn.Module,
        hf_path: str,
        cs_path: str,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> None:
        """Copy constant-initialized leaves below one mapped model section."""
        local_specs = getattr(self, "_local_tp_shard_specs", {})
        for rel_name, child in cs_module.named_modules():
            if hasattr(child, "reset_parameters"):
                continue
            source_rel = self._global_repeated_layer_name(cs_path, rel_name)
            for param_name, param in child.named_parameters(recurse=False):
                local_key = self._join_parameter_path(cs_path, rel_name, param_name)
                if local_key in local_specs:
                    continue
                source_key = self._join_parameter_path(
                    hf_path, source_rel, param_name
                )
                try:
                    source = hf_model.get_parameter(source_key)
                except AttributeError:
                    continue
                replacement = source.to(
                    device=self._materialization_device_for_tensor(local_key, device),
                    dtype=dtype or source.dtype,
                )
                replacement = self._fit_constant_parameter(
                    replacement, param, child, source_key
                )
                self._set_tensor(local_key, replacement, param.requires_grad)

    @staticmethod
    def _join_parameter_path(root: str, relative: str, parameter: str) -> str:
        return f"{root}.{relative}.{parameter}" if relative else f"{root}.{parameter}"

    @staticmethod
    def _fit_constant_parameter(
        replacement: torch.Tensor,
        target: nn.Parameter,
        owner: nn.Module,
        source_key: str,
    ) -> torch.Tensor:
        """Fit a constant source to a manually row-sharded parameter shape."""
        if replacement.shape == target.shape:
            return replacement
        tp_size = int(getattr(owner, "_cornstarch_tp_size", 1))
        tp_rank = int(getattr(owner, "_cornstarch_tp_rank", 0))
        can_shard_rows = (
            tp_size > 1
            and replacement.ndim > 0
            and replacement.shape[0] % tp_size == 0
            and replacement.shape[1:] == target.shape[1:]
        )
        if can_shard_rows:
            return replacement.chunk(tp_size, dim=0)[tp_rank].contiguous()
        raise RuntimeError(
            f"Cannot copy constant parameter {source_key!r} with shape "
            f"{tuple(replacement.shape)} into distributed shape {tuple(target.shape)}."
        )

    def _copy_local_tp_parameters(
        self, device: torch.device, dtype: torch.dtype | None
    ) -> None:
        """Initialize manually section-sharded TP tensors from full HF values."""
        local_specs: dict[str, tuple] = getattr(
            self, "_local_tp_shard_specs", {}
        )
        if not local_specs:
            return
        hf_model = self._hf_model_factory(copy.deepcopy(self.hf_config))
        if dtype is None:
            hf_model.to(device=device)
        else:
            hf_model.to(device=device, dtype=dtype)
        _, layers_name, _ = self._section_names()
        for local_name, spec in local_specs.items():
            source_name = local_name
            prefix = f"{layers_name}."
            if local_name.startswith(prefix):
                suffix = local_name[len(prefix):]
                source_suffix = self._global_repeated_layer_name(
                    layers_name, suffix
                )
                source_name = f"{prefix}{source_suffix}"
            hf_name = self._state_mapper.cornstarch_to_hf_key(source_name)
            source = hf_model.get_parameter(hf_name)
            local = self._slice_local_tp_tensor(source, spec).to(
                device=self._materialization_device_for_tensor(local_name, device),
                dtype=dtype if dtype is not None and source.is_floating_point() else None,
            )
            target = self.get_parameter(local_name)
            self._set_tensor(local_name, local, target.requires_grad)
        del hf_model

    def _global_repeated_layer_name(self, section_path: str, name: str) -> str:
        """Translate a PP-local repeated-layer path back to its HF global index."""
        _, layers_name, _ = self._section_names()
        if section_path != layers_name or not name:
            return name
        first, separator, remainder = name.partition(".")
        if not first.isdigit():
            return name
        global_index = int(first) + int(getattr(self, "_pipeline_layer_offset", 0))
        return f"{global_index}.{remainder}" if separator else str(global_index)

    def _global_cornstarch_key(self, local_key: str) -> str:
        """Map a PP-local state key back to the global Cornstarch topology."""
        _, layers_name, _ = self._section_names()
        prefix = f"{layers_name}."
        if not local_key.startswith(prefix):
            return local_key
        suffix = self._global_repeated_layer_name(
            layers_name, local_key[len(prefix):]
        )
        return f"{prefix}{suffix}"

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
