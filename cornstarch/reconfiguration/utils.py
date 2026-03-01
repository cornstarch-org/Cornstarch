"""Utility functions for dynamic reconfiguration."""

from typing import Dict, List, Set

import torch
import torch.nn as nn


def get_all_param_names(
    source_ownership: Dict[int, 'LayerOwnership'],
    target_ownership: Dict[int, 'LayerOwnership']
) -> Set[str]:
    """Get all unique parameter names from source and target ownership.

    Args:
        source_ownership: Source ownership mapping
        target_ownership: Target ownership mapping

    Returns:
        Set of all parameter names
    """
    all_names = set()

    for ownership in source_ownership.values():
        all_names.update(ownership.layer_names)
        all_names.update(ownership.is_placeholder.keys())

    for ownership in target_ownership.values():
        all_names.update(ownership.layer_names)
        all_names.update(ownership.is_placeholder.keys())

    return all_names


def get_tp_group_ranks(rank: int, tp_size: int) -> List[int]:
    """Get all ranks in the same TP group as the given rank.

    Assumes ranks are organized sequentially by TP groups:
    - Ranks [0, 1, ..., tp_size-1] form TP group 0
    - Ranks [tp_size, tp_size+1, ..., 2*tp_size-1] form TP group 1
    - etc.

    Args:
        rank: The rank to query
        tp_size: Tensor parallelism size

    Returns:
        List of ranks in the same TP group
    """
    tp_group_id = rank // tp_size
    start_rank = tp_group_id * tp_size
    return list(range(start_rank, start_rank + tp_size))


def get_tp_rank(rank: int, tp_size: int) -> int:
    """Get the TP rank (position within TP group) for a given global rank.

    Args:
        rank: Global rank
        tp_size: Tensor parallelism size

    Returns:
        TP rank (0 to tp_size-1)
    """
    return rank % tp_size


def get_pp_stage(rank: int, tp_size: int) -> int:
    """Get the PP stage ID for a given global rank.

    Args:
        rank: Global rank
        tp_size: Tensor parallelism size

    Returns:
        PP stage ID
    """
    return rank // tp_size


def get_ranks_for_pp_stage(stage_id: int, tp_size: int) -> List[int]:
    """Get all ranks belonging to a specific PP stage.

    Args:
        stage_id: Pipeline stage ID
        tp_size: Tensor parallelism size

    Returns:
        List of ranks in the PP stage
    """
    start_rank = stage_id * tp_size
    return list(range(start_rank, start_rank + tp_size))


def get_param_by_name(model: nn.Module, param_name: str) -> nn.Parameter:
    """Get a parameter by its full name.

    Args:
        model: The model
        param_name: Full parameter name (e.g., "layers.0.self_attn.q_proj.weight")

    Returns:
        The parameter

    Raises:
        KeyError: If parameter not found
    """
    for name, param in model.named_parameters():
        if name == param_name:
            return param
    raise KeyError(f"Parameter {param_name} not found in model")


def set_param_by_name(model: nn.Module, param_name: str, value: torch.Tensor) -> None:
    """Set a parameter by its full name.

    Preserves the existing ``nn.Parameter`` object's identity when the slot
    already holds a parameter, updating only its ``.data``.  This keeps
    optimizer ``param_groups`` / ``state`` keys valid across reconfiguration.
    A new ``nn.Parameter`` is created only when the slot was previously a
    placeholder (``None``) or did not exist.

    Args:
        model: The model
        param_name: Full parameter name
        value: New parameter value
    """
    parts = param_name.split('.')
    module = model

    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    param_local_name = parts[-1]
    existing = getattr(module, param_local_name, None)

    if isinstance(existing, nn.Parameter):
        # In-place update: preserves object identity so the optimizer's
        # param_groups and state dict keys remain valid.
        existing.data = value
    else:
        # Slot was a placeholder (None) or absent — create a fresh Parameter.
        if hasattr(module, param_local_name):
            delattr(module, param_local_name)
        module.register_parameter(param_local_name, nn.Parameter(value))


def get_parent_module(model: nn.Module, param_name: str) -> nn.Module:
    """Get the parent module containing a parameter.

    Args:
        model: The model
        param_name: Full parameter name

    Returns:
        The parent module containing the parameter
    """
    parts = param_name.split('.')
    module = model

    # Navigate to the parent module
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module
