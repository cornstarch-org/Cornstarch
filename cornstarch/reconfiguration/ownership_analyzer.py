"""Analyzer for determining tensor ownership across ranks."""

from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn

from .data_structures import LayerOwnership
from .utils import get_parent_module


class TensorOwnershipAnalyzer:
    """Analyzes current layer ownership in a parallelized model."""

    def __init__(self, model: nn.Module, plugin=None):
        """Initialize the analyzer.

        Args:
            model: The parallelized model
            plugin: The parallel plugin (optional, for future use)
        """
        self.model = model
        self.plugin = plugin
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def analyze(self) -> Dict[int, LayerOwnership]:
        """Analyze ownership across all ranks.

        Returns:
            Dictionary mapping rank to LayerOwnership
        """
        # Get local ownership
        local_ownership = self._get_local_ownership()

        # Gather from all ranks
        all_ownership = self._gather_ownership(local_ownership)

        return all_ownership

    def _get_local_ownership(self) -> LayerOwnership:
        """Get layers held by this rank.

        Returns:
            LayerOwnership for this rank
        """
        layer_names = []
        is_placeholder = {}

        for name, param in self.model.named_parameters():
            # Check if placeholder
            if self._is_placeholder(name, param):
                is_placeholder[name] = True
            else:
                layer_names.append(name)
                is_placeholder[name] = False

        return LayerOwnership(
            rank=self.rank,
            layer_names=layer_names,
            is_placeholder=is_placeholder
        )

    def _is_placeholder(self, name: str, param) -> bool:
        """Check if a parameter is a TensorPlaceholder.

        Args:
            name: Parameter name
            param: Parameter value

        Returns:
            True if parameter is a placeholder
        """
        # Import here to avoid circular dependencies
        from cornstarch.shardformer.shard.placeholder import TensorPlaceholder

        # Method 1: Check type directly
        if isinstance(param, TensorPlaceholder):
            return True

        # Method 2: Check if parameter is None (after placeholder conversion)
        if param is None:
            return True

        # Method 3: Check module's _parameter_placeholders attribute
        try:
            parent_module = get_parent_module(self.model, name)
            if hasattr(parent_module, '_parameter_placeholders'):
                param_local_name = name.split('.')[-1]
                if param_local_name in parent_module._parameter_placeholders:
                    return True
        except Exception:
            # If we can't find the parent module, assume it's not a placeholder
            pass

        return False

    def _gather_ownership(
        self,
        local_ownership: LayerOwnership
    ) -> Dict[int, LayerOwnership]:
        """Gather ownership info from all ranks.

        Args:
            local_ownership: Ownership for this rank

        Returns:
            Dictionary mapping all ranks to their ownership
        """
        if not dist.is_initialized():
            # Not in distributed mode, return only local
            return {self.rank: local_ownership}

        world_size = dist.get_world_size()

        # Serialize local ownership to dict for all_gather_object
        local_data = {
            'rank': local_ownership.rank,
            'layer_names': local_ownership.layer_names,
            'is_placeholder': local_ownership.is_placeholder
        }

        # All-gather
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)

        # Reconstruct LayerOwnership objects
        all_ownership = {}
        for data in gathered_data:
            if data is not None:
                ownership = LayerOwnership(
                    rank=data['rank'],
                    layer_names=data['layer_names'],
                    is_placeholder=data['is_placeholder']
                )
                all_ownership[data['rank']] = ownership

        return all_ownership
