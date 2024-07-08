from typing import List, Union

import torch.distributed as dist
from colossalai.lazy import LazyInitContext
from colossalai.shardformer.layer.parallel_module import ParallelModule
from colossalai.shardformer.layer.qkv_fused_linear import (
    FusedLinear1D_Col as ColossalFusedLinear1D_Col,
)
from torch import nn

"""
Temporarily applies bugfix in FusedLinear1D_Col ([#5894](https://github.com/hpcaitech/ColossalAI/pull/5894))
"""


class FusedLinear1D_Col(ColossalFusedLinear1D_Col):
    @staticmethod
    def from_native_module(
        module: nn.Module,
        process_group: Union[dist.ProcessGroup, List[dist.ProcessGroup]],
        n_fused: int,
        *args,
        **kwargs,
    ) -> ParallelModule:
        r"""
        Convert a fused `torch.nn.linear` layer to a parallelized linear layer.

        Args:
            module (`nn.Linear`): The module to be converted.
            process_group (`Union[ProcessGroup, List[ProcessGroup]]`): The process group to be used for weight sharding and communication.
            n_fused (int): The number of layers to be fused. In common, Q,K,V are fused in one weight.
        """
        LazyInitContext.materialize(module)

        # get the attributes
        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device

        # ensure only one process group is passed
        if isinstance(process_group, (list, tuple)):
            assert (
                len(process_group) == 1
            ), f"Expected only one process group, got {len(process_group)}."
            process_group = process_group[0]

        linear_1d = ColossalFusedLinear1D_Col(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            n_fused=n_fused,
            *args,
            **kwargs,
        )

        return linear_1d
