"""Data structures for dynamic parallel reconfiguration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class LayerOwnership:
    """Describes which model layers a rank owns and what portion of each.

    For parameters that are TP-sharded, ``shard_range`` records the
    [start, end) interval of the full parameter tensor that this rank holds
    along ``shard_dim``.  ``None`` means the rank holds the complete tensor
    (no TP sharding for that parameter).

    Attributes:
        rank:          The rank ID.
        layer_names:   Parameter names held (fully or partially) by this rank.
        is_placeholder: True when the name is a placeholder (not actually held).
        shard_range:   param_name → (start, end) in the full tensor, or None.
        shard_dim:     param_name → which axis is sharded, or None.
    """
    rank: int
    layer_names: List[str]
    is_placeholder: Dict[str, bool] = field(default_factory=dict)
    shard_range: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    shard_dim: Dict[str, Optional[int]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.is_placeholder:
            self.is_placeholder = {name: False for name in self.layer_names}


@dataclass
class AllToAllPlan:
    """All-to-all communication plan for a single parameter.

    Attributes:
        param_name: Name of the parameter
        send_to_ranks: Ranks that need this parameter (current rank will send to them)
        recv_from_ranks: Ranks that have this parameter (current rank will receive from them)
        tensor_shape: Shape of the tensor
        tensor_dtype: Data type of the tensor
    """
    param_name: str
    send_to_ranks: List[int]
    recv_from_ranks: List[int]
    tensor_shape: torch.Size
    tensor_dtype: torch.dtype
