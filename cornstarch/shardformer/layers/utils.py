from flash_attn import flash_attn_qkvpacked_func
from contextlib import contextmanager
from typing import List, Optional, Union
from enum import Enum, auto

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import inspect
from functools import cache


__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]

import torch
import torch.distributed as dist
from torch import nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed import ProcessGroup, get_world_size

class SplitMode(Enum):
    ZIGZAG = auto()
    UNIFORM = auto()

# code adapted from: https://github.com/hpcaitech/ColossalAI/blob/dcd41d09730854f2f7d58f6b5cc5096617a6c2f3/colossalai/shardformer/layer/utils.py#L294
def split_batch_zigzag(
    batch: Union[torch.Tensor, List[torch.Tensor]], sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Split the input along the sequence dimension for Ring Attention. Naively spliting the attention mask
    in the causal setting will result in the preceding ranks having much less workload.
    We split after "folding" the 2D attention mask in half (https://github.com/zhuzilin/ring-flash-attention/issues/2).
    For example, for sp_size = 4 and seq_len = 8, we get | s0, s7 | s1, s6 | s2, s5 | s3, s4 |.

    Args:
        batch (List[torch.Tensor] or Tensor): The input tensor(s) to split.
        sp_group (ProcessGroup): The process group for sequence parallelism.
        seq_dim (int): The sequence dimension to split.
        is_label (bool): If True, mask and shift the tensor for next token prediction.

    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size == 1:
        return batch

    if isinstance(batch, torch.Tensor):
        batch = [batch]
    seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1

    if sp_size > 1:
        for idx, tensor in enumerate(batch):
            assert (
                tensor.shape[seq_dim] // (sp_size * 2) > 1 and tensor.shape[seq_dim] % (sp_size * 2) == 0
            ), f"Bro, the seq length {tensor.shape[seq_dim]} for tensor {idx} can't be split by {sp_size * 2}!"
            if is_label:
                assert tensor.dim() == 2, "Label shape should be (B, Seqlen)"
                tensor = torch.cat([tensor[:, 1:], torch.full_like(tensor[:, :1], -100)], dim=1)
                # @runyu, what is the meaning of set the first token to -100?

            tensor = tensor.view(
                *tensor.shape[:seq_dim],
                2 * sp_size,
                tensor.shape[seq_dim] // (2 * sp_size),
                *tensor.shape[seq_dim + 1 :],
            )
            indices = torch.tensor([sp_rank, 2 * sp_size - 1 - sp_rank], device=tensor.device)
            tensor = tensor.index_select(seq_dim, indices).contiguous()
            # (B, 2, Sq // (2 * sp_size), ...) -> (B, Sq // sp_size, ...)
            batch[idx] = tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])

    if len(batch) == 1:
        return batch[0]
    return batch


def split_batch_uniform(
    batch: torch.Tensor, sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> torch.Tensor:

    """
    split them evenly by seq_dim
    """

    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size == 1:
        return batch

    seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1

    seq_len = batch.shape[seq_dim]

    assert seq_len % sp_size == 0, f"Sequence length {seq_len} must be divisible by {sp_size}!"
    split_batch = batch.chunk(sp_size, dim=seq_dim)[sp_rank].contiguous()

    return split_batch


def split_batch_anymask(
    batch: torch.Tensor, sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> torch.Tensor:
    """
    split them using some strategy
    TODO(@insu) implement this
    """
    return NotImplementedError("Not implemented yet for anymask automatically split")


@cache
def get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group
        )
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []



def split_attn_mask(
    batch: Union[torch.Tensor, List[torch.Tensor]], 
    sp_group: ProcessGroup, 
    seq_dim: int = 1, 
    is_label: bool = False, 
    split_mode: SplitMode = SplitMode.ZIGZAG
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if split_mode == SplitMode.ZIGZAG:
        return split_batch_zigzag(batch, sp_group, seq_dim, is_label)
    elif split_mode == SplitMode.UNIFORM:
        return split_batch_uniform(batch, sp_group, seq_dim, is_label)
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
