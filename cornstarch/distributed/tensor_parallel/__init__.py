"""Tensor parallelism via PyTorch DTensor.

Tensor parallelism shards each repeated layer's linear weights across the TP
ranks: attention/FFN input projections are column-parallel and output
projections are row-parallel, so a forward pass needs one all-reduce per
attention and per FFN block.  The split is described by a per-model-family
plan in ``plans.py`` (submodule path -> ``ParallelStyle``) and applied with
``torch.distributed.tensor.parallel.parallelize_module``.

``apply_tensor_parallel`` runs on the meta module before ``materialize()``:
``parallelize_module`` records the DTensor sharding spec on the meta
parameters, and real (already-sharded) storage is allocated during
materialization.

DTensor is used *only* for TP weight sharding here — it is deliberately not
extended to pipeline parallelism or FSDP-style data parallelism (those use the
explicit Megatron/ColossalAI-style schedule and manual gradient all-reduce
instead).

Unlike the earlier trial, a model family with no registered TP plan raises
``TensorParallelNotSupportedError`` rather than silently leaving the model
unsharded — a silent no-op would let a caller believe a model is tensor
parallel when it is not, producing wrong sharding math downstream.
"""
from __future__ import annotations

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module

import torch
from torch import nn

from cornstarch.distributed.tensor_parallel.plans import (
    get_layer_tp_plan,
    get_tp_plan,
)
from cornstarch.models.model_base import CornstarchModelBase


class TensorParallelNotSupportedError(NotImplementedError):
    """Raised when ``apply_tensor_parallel`` has no plan for a model family."""


def apply_tensor_parallel(module: CornstarchModelBase, tp_mesh: DeviceMesh) -> None:
    """Wrap linear layers in each repeated layer with DTensor column/row sharding.

    Must be called before ``module.materialize()`` so the sharding spec is
    recorded on meta parameters; actual storage is allocated during
    materialization on the correct device.

    Raises
    ------
    TensorParallelNotSupportedError
        If no TP plan is registered for the module's model family.  This is an
        explicit error (not a silent no-op) so callers cannot mistake an
        unsharded model for a tensor-parallel one.
    """
    config_class_name = type(module.hf_config).__name__
    plan = get_tp_plan(config_class_name)
    if plan is None:
        raise TensorParallelNotSupportedError(
            f"No tensor-parallel plan is registered for model family "
            f"'{config_class_name}'. Register a per-layer TP plan in "
            f"cornstarch/distributed/tensor_parallel/plans.py before applying "
            f"tensor parallelism to this model."
        )

    _, layers_name, _ = module._section_names()
    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    local_specs: dict[str, tuple] = getattr(module, "_local_tp_shard_specs", {})
    for layer_index, layer in enumerate(getattr(module, layers_name)):
        layer_plan = get_layer_tp_plan(
            config_class_name, getattr(layer, "layer_type", None)
        )
        assert layer_plan is not None
        if getattr(layer, "layer_type", None) == "linear_attention":
            _validate_gated_delta_tp(layer.linear_attn, tp_size)
        parallelize_module(layer, tp_mesh, layer_plan)
        if getattr(layer, "layer_type", None) == "linear_attention":
            relative_specs = _shard_gated_delta_state(
                layer.linear_attn, tp_rank, tp_size
            )
            local_specs.update(
                {
                    f"{layers_name}.{layer_index}.linear_attn.{name}": spec
                    for name, spec in relative_specs.items()
                }
            )
    module._local_tp_shard_specs = local_specs


def _validate_gated_delta_tp(linear_attn: nn.Module, tp_size: int) -> None:
    """Validate the semantic GDN axes used by any compatible converted layer."""
    fields = {
        "linear_num_key_heads": int(linear_attn.num_k_heads),
        "linear_num_value_heads": int(linear_attn.num_v_heads),
        "linear key channels": int(linear_attn.key_dim),
        "linear value channels": int(linear_attn.value_dim),
    }
    invalid = {name: size for name, size in fields.items() if size % tp_size}
    if invalid:
        details = ", ".join(f"{name}={size}" for name, size in invalid.items())
        raise ValueError(
            f"Gated DeltaNet TP size {tp_size} does not divide {details}."
        )


def _shard_gated_delta_state(
    linear_attn: nn.Module, tp_rank: int, tp_size: int
) -> dict[str, tuple]:
    """Shard GDN state and convolution channels by semantic Q/K/V sections.

    DTensor handles ordinary projection matrices declaratively. These tensors
    need local replacement because a fused Q/K/V tensor contains unequal
    semantic sections and the recurrent scalars/depthwise convolution follow
    those head sections. The operation depends on the unified layer contract,
    not on the original Hugging Face root model family.
    """
    linear_attn._cornstarch_tp_rank = tp_rank
    linear_attn._cornstarch_tp_size = tp_size
    specs: dict[str, tuple] = {}

    qkv = linear_attn.in_proj_qkv.weight
    section_sizes = (
        int(linear_attn.key_dim),
        int(linear_attn.key_dim),
        int(linear_attn.value_dim),
    )
    qkv_sections = qkv.split(section_sizes, dim=0)
    local_qkv = torch.cat(
        [section.chunk(tp_size, dim=0)[tp_rank] for section in qkv_sections],
        dim=0,
    ).clone()
    linear_attn.in_proj_qkv.weight = nn.Parameter(
        local_qkv, qkv.requires_grad
    )
    linear_attn.in_proj_qkv.out_features = local_qkv.shape[0]
    specs["in_proj_qkv.weight"] = (
        "sections",
        0,
        section_sizes,
        tp_rank,
        tp_size,
    )

    for name in ("A_log", "dt_bias"):
        parameter = getattr(linear_attn, name)
        local = parameter.chunk(tp_size, dim=0)[tp_rank].clone()
        setattr(linear_attn, name, nn.Parameter(local, parameter.requires_grad))
        specs[name] = ("chunk", 0, tp_rank, tp_size)

    weight = linear_attn.conv1d.weight
    # Conv channels follow fused [Q; K; V] order. Select this TP lane from
    # every semantic section rather than taking one contiguous third.
    sections = weight.split(section_sizes, dim=0)
    local_weight = torch.cat(
        [section.chunk(tp_size, dim=0)[tp_rank] for section in sections], dim=0
    ).clone()
    linear_attn.conv1d.weight = nn.Parameter(
        local_weight, weight.requires_grad
    )
    specs["conv1d.weight"] = (
        "sections",
        0,
        section_sizes,
        tp_rank,
        tp_size,
    )
    if linear_attn.conv1d.bias is not None:
        bias_sections = linear_attn.conv1d.bias.split(section_sizes, dim=0)
        local_bias = torch.cat(
            [section.chunk(tp_size, dim=0)[tp_rank] for section in bias_sections]
        ).clone()
        linear_attn.conv1d.bias = nn.Parameter(
            local_bias, linear_attn.conv1d.bias.requires_grad
        )
        specs["conv1d.bias"] = (
            "sections",
            0,
            section_sizes,
            tp_rank,
            tp_size,
        )

    linear_attn.num_k_heads //= tp_size
    linear_attn.num_v_heads //= tp_size
    linear_attn.key_dim //= tp_size
    linear_attn.value_dim //= tp_size
    linear_attn.conv_dim //= tp_size
    linear_attn.conv1d.in_channels = linear_attn.conv_dim
    linear_attn.conv1d.out_channels = linear_attn.conv_dim
    linear_attn.conv1d.groups = linear_attn.conv_dim
    return specs
