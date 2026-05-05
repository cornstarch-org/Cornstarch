from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class RepeatedLayerStack(nn.Module):
    def __init__(self, layers: Iterable[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, index: int) -> nn.Module:
        return self.layers[index]

    def offload_to_cpu(self, layer_indices: Iterable[int] | None = None) -> None:
        indices = range(len(self.layers)) if layer_indices is None else layer_indices
        for index in indices:
            layer = self.layers[index]
            if not any(param.is_meta for param in layer.parameters(recurse=True)):
                layer.to("cpu")

    def materialize_layers(self, device: str | torch.device) -> None:
        device = torch.device(device)
        for layer in self.layers:
            tensors = list(layer.parameters(recurse=True)) + list(layer.buffers(recurse=True))
            if any(tensor.is_meta for tensor in tensors):
                layer.to_empty(device=device)
            else:
                layer.to(device)
