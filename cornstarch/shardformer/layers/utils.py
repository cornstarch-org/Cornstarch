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

def split_batch(
    batch: torch.Tensor, sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False, sp_mode: str = "ring_attn"
) -> torch.Tensor:
    if sp_mode == "ring_attn" or sp_mode == "ring_attn_uniform":
        return split_batch_uniform(batch, sp_group, seq_dim, is_label)
    elif sp_mode == "ring_attn_zig_zag":
        return split_batch_zig_zag(batch, sp_group, seq_dim, is_label)
    elif sp_mode == "ring_attn_optimal":
        return split_batch_optimal(batch, sp_group, seq_dim, is_label)
    else:
        raise ValueError(f"Unknown split mode: {sp_mode}")

def split_batch_zig_zag(
    batch: torch.Tensor, sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> torch.Tensor:
    """
    split them using zigzag strategy
    """
    sp_size = dist.get_world_size(sp_group)
    sp_rank = dist.get_rank(sp_group)
    if sp_size == 1:
        return batch

    seq_dim = seq_dim if seq_dim != -1 else batch[0].dim() - 1

    seq_len = batch.shape[seq_dim]

    assert seq_len % (sp_size * 2) == 0, f"Sequence length {seq_len} must be divisible by {sp_size * 2}!"

    tensor = batch.view(
        *batch.shape[:seq_dim],
        2 * sp_size,
        batch.shape[seq_dim] // (2 * sp_size),
        *batch.shape[seq_dim + 1 :],
    )
    indices = torch.tensor([sp_rank, 2 * sp_size - 1 - sp_rank], device=tensor.device)
    tensor = tensor.index_select(seq_dim, indices).contiguous()
    # (B, 2, Sq // (2 * sp_size), ...) -> (B, Sq // sp_size, ...)
    batch = tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])

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


def split_batch_optimal(
    batch: torch.Tensor, sp_group: ProcessGroup, seq_dim: int = 1, is_label: bool = False
) -> torch.Tensor:
    """
    split them using some strategy
    TODO(@insu) implement this
    """
    raise NotImplementedError("Not implemented yet for anymask automatically split")


@cache
def get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args

def update_attention_mask(attention_mask, head_num, head_dim=1):
    """
    transform attention mask 3d(B, L, L) to 4d(B, H, L, L)
    An example:
    attention_mask = torch.tensor([[1, 0, 0], [1, 0, 1], [0, 1, 1]])
    update_attention_mask(attention_mask, head_num=2, head_dim=1)
    >>> torch.tensor([[[1, 0, 0], [1, 0, 1], [0, 1, 1]], [[1, 0, 0], [1, 0, 1], [0, 1, 1]]])
    """
    return attention_mask.unsqueeze(dim=head_dim).expand(-1, head_num, -1, -1)

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)

    if block_lse.shape[:3] != block_out.shape[:3]:
        # NOTE(@runyu): flash attn by tridao need transpose the last two dims
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        # print("tridao's flash attn")
    else:
        block_lse = block_lse.unsqueeze(dim=-1)

    # print(f"shape of block_out: {block_out.shape}, shape of block_lse: {block_lse.shape}")
    # print(f"shape of out: {out.shape}, shape of lse: {lse.shape}")

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
        if block_lse.shape[:3] != block_out.shape[:3]:
            lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
            print("tridao's flash attn")
        else:
            lse = block_lse.unsqueeze(dim=-1)
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
