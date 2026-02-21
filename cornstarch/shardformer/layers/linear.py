from typing import List, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from colossalai.shardformer.layer import Linear1D_Row
from colossalai.shardformer.layer._operation import reducescatter_forward_gather_backward
from colossalai.shardformer.layer.utils import is_share_sp_tp
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter

__all__ = ["Linear1D_Row_ReduceScatter"]


class Linear1D_Row_ReduceScatter(Linear1D_Row):
    """Row-parallel linear layer that scatters the output along the hidden
    (last) dimension instead of performing an all-reduce.

    In the standard ``Linear1D_Row`` forward pass the partial products from
    each rank are all-reduced, yielding an identical ``(seq, H)`` tensor on
    every TP rank.  This subclass replaces that all-reduce with a
    reduce-scatter along ``dim=-1`` so that rank *j* receives the shard
    ``output[:, j*H//tp : (j+1)*H//tp]``.  The corresponding backward
    performs an all-gather to reconstruct the full gradient.

    This is useful at the encoder→LLM border when ``enc_tp > llm_tp``:
    each encoder TP rank emits a unique hidden shard, eliminating the
    duplicated transfers that would otherwise be discarded by the LLM.
    """

    @staticmethod
    def from_native_module(
        module,
        process_group: Union[ProcessGroup, List[ProcessGroup]],
        **kwargs,
    ) -> "Linear1D_Row_ReduceScatter":
        from colossalai.lazy import LazyInitContext

        LazyInitContext.materialize(module)

        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device

        if isinstance(process_group, (list, tuple)):
            assert len(process_group) == 1, (
                f"Expected only one process group, got {len(process_group)}."
            )
            process_group = process_group[0]

        tp_size = dist.get_world_size(process_group)
        if in_features < tp_size:
            return module

        if in_features % tp_size != 0:
            raise ValueError(
                f"in_features={in_features} is not divisible by tp_size={tp_size}."
            )

        return Linear1D_Row_ReduceScatter(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            process_group=process_group,
            weight=module.weight,
            bias_=module.bias,
            **kwargs,
        )

    def forward(self, input_: Tensor) -> Tensor:
        # Handle input splitting (same as Linear1D_Row).
        if self.parallel_input:
            assert input_.shape[-1] == self.weight.shape[-1], (
                "Invalid shapes in Linear1D_Row_ReduceScatter forward: "
                f"input={input_.shape}, weight={self.weight.shape}. "
                f"Expected last dim of input {self.weight.shape[-1]}."
            )
        else:
            from colossalai.nn.layer.utils import divide
            from colossalai.shardformer.layer._operation import (
                split_forward_gather_backward,
            )

            assert divide(input_.shape[-1], self.num_partitions) == self.weight.shape[-1], (
                "Invalid shapes in Linear1D_Row_ReduceScatter forward: "
                f"input={input_.shape}, weight={self.weight.shape}."
            )
            input_ = split_forward_gather_backward(
                input_,
                dim=-1,
                process_group=self.process_group,
                fp8_communication=self.fp8_communication,
            )

        if self.stream_chunk_num > 1:
            raise RuntimeError(
                "Linear1D_Row_ReduceScatter does not support stream_chunk_num > 1."
            )

        if is_share_sp_tp(self.seq_parallel_mode):
            # SP mode already uses reduce-scatter internally; fall back to
            # parent behaviour (seq dim scatter, not hidden dim scatter).
            from colossalai.shardformer.layer._operation import (
                linear_reducescatter_forward_gather_backward,
            )

            output = linear_reducescatter_forward_gather_backward(
                input_,
                self.weight,
                process_group=self.process_group,
                dim=self.seq_parallel_dim,
                ring=self.seq_parallel_mode == "ring",
            )
        else:
            output_parallel = F.linear(input_, self.weight)
            output = reducescatter_forward_gather_backward(
                output_parallel,
                self.process_group,
                dim=-1,
                fp8_communication=self.fp8_communication,
            )

        if not self.skip_bias_add:
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return output, self.bias
