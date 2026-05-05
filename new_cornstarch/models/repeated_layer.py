from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class RepeatedLayerStack(nn.Module):
    """Wrap repeated transformer blocks with layer-wise memory helpers."""

    def __init__(self, layers: Iterable[nn.Module]):
        """Store the provided layers as a registered PyTorch module list."""
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def __len__(self) -> int:
        """Return the number of repeated layers in the stack."""
        return len(self.layers)

    def __iter__(self):
        """Iterate over the repeated layers in their execution order."""
        return iter(self.layers)

    def __getitem__(self, index: int) -> nn.Module:
        """Return a repeated layer by integer index."""
        return self.layers[index]

    def offload_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        """Move selected materialized layers to CPU without touching meta layers."""
        indices = range(len(self.layers)) if layer_indices is None else layer_indices
        for index in indices:
            layer = self.layers[index]
            if not any(param.is_meta for param in layer.parameters(recurse=True)):
                layer.to("cpu")

    def materialize_layers(self, device: str | torch.device) -> None:
        """Allocate or move all repeated layers onto the requested device."""
        device = torch.device(device)
        for layer in self.layers:
            tensors = list(layer.parameters(recurse=True)) + list(layer.buffers(recurse=True))
            if any(tensor.is_meta for tensor in tensors):
                layer.to_empty(device=device)
            else:
                layer.to(device)
