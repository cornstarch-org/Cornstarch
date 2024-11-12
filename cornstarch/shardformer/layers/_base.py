from typing import Optional

import torch
import torch.distributed as dist


class RingAttentionBase(torch.autograd.Function):
    """
    Base class for ring attention.
    All RingAttention implementation should inherit from this class
    and implement abstract methods to be used in models.
    """

    @staticmethod
    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sp_group: dist.ProcessGroup,
        dropout_p: Optional[float] = 0.0,
        softmax_scale: Optional[float] = None,
        deterministic: Optional[bool] = False,
        return_softmax: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
