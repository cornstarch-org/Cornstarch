import torch
from .utils import RingComm, get_default_args, update_out_and_lse
from flash_attn import flash_attn_qkvpacked_func
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from cornstarch.kernel.interface import _flash_attn_anymask_forward, _flash_attn_anymask_backward
from typing import Callable

def ring_flash_attn_anymask_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale,
    ATTN_IMPL: Callable = _flash_attn_anymask_forward,
    dropout_p=0,
    causal=True,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            params = get_default_args(ATTN_IMPL).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "mask": mask,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "window_size_left": window_size_left,
                    "window_size_right": window_size_right,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            block_out, block_lse, _, _ = ATTN_IMPL(**params)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    ATTN_IMPL: Callable = _flash_attn_forward,
    dropout_p=0,
    causal=True,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            params = get_default_args(ATTN_IMPL).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal and step == 0,
                    "window_size_left": window_size_left,
                    "window_size_right": window_size_right,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                }
            )
            block_out, block_lse, _, _ = ATTN_IMPL(**params)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()
        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
                    "window_size_left": window_size_left,
                    "window_size_right": window_size_right,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            _flash_attn_backward(**params)

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk = next_dk
            dv = next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)

def ring_flash_attn_anymask_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    mask,
    ATTN_IMPL: Callable = _flash_attn_anymask_backward,
    dropout_p=0,
    window_size_left=-1,
    window_size_right=-1,
    alibi_slopes=None,
    deterministic=False,
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None

    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            kv_comm.commit()
        if step <= kv_comm.rank:
            params = get_default_args(ATTN_IMPL).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "mask": mask,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "window_size_left": window_size_left,
                    "window_size_right": window_size_right,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )
            ATTN_IMPL(**params)

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv
        elif step != 0:
            d_kv_comm.wait()
            dk = next_dk
            dv = next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()

    d_kv_comm.wait()

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)

