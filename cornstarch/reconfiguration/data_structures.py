"""Data structures for dynamic parallel reconfiguration."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class LayerOwnership:
    """Describes which model layers a rank owns.

    Attributes:
        rank: The rank ID
        layer_names: List of parameter names held by this rank
        is_placeholder: Mapping from parameter name to whether it's a placeholder
    """
    rank: int
    layer_names: List[str]
    is_placeholder: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize is_placeholder dict if not provided."""
        if not self.is_placeholder:
            # By default, all layer_names are not placeholders
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
