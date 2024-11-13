import torch
from typing import Optional, Tuple


def _attn_anymask_forward(
    ctx: torch.autograd.function.FunctionCtx,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    softmax_scale: float = 1.0,
    dropout_p: float = 0.0,  # TODO(@runyu) not used
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_softmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    p = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    # p[:, :, mask == 0] = float("-inf")
    num_heads = q.shape[1]
    index = mask == 0
    index = index.unsqueeze(dim=1).expand(-1, num_heads, -1, -1)
    p[index] = -1e4 if q.dtype == torch.float16 else -1e9
    softmax_lse = torch.logsumexp(p.float(), dim=-1)
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    out = torch.matmul(p, v)

    return out, softmax_lse, None, None


def _attn_anymask_backward(
    ctx: torch.autograd.function.FunctionCtx,
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    mask_info: Tuple[torch.Tensor],
    dropout_p: float = 0.0,
    softmax_scale: float = 1.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    rng_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (grad_output, output, attn_probs, scores, Q_h, K_h, V_h)

    grad_softmax_lse, mask = mask_info

    num_heads = q.shape[1]
    mask = mask.unsqueeze(dim=1).expand(-1, num_heads, -1, -1)

    # Step 1: recompute attn_probs
    grad_attn_output = dout  # [batch_size, n_heads, seq_len, d_head]
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
    # scores[mask == 0] = float("-inf")
    scores[mask == 0] = -1e4 if scores.dtype == torch.float16 else -1e9
    # attn_probs = torch.softmax(scores, dim=-1) # [batch_size, n_heads, seq_len, seq_len]
    if softmax_lse is not None:
        attn_probs = torch.exp(scores - softmax_lse.unsqueeze(-1)).to(scores.dtype)
    else:
        attn_probs = torch.softmax(
            scores, dim=-1
        )  # [batch_size, n_heads, seq_len, seq_len]

    # Step 2: Gradient w.r.t. V
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_V = torch.matmul(attn_probs.transpose(-2, -1), grad_attn_output)
    if dv is None:
        dv = grad_V
    else:
        dv.copy_(grad_V)

    # Step 3: Gradient w.r.t. attention probabilities
    # [batch_size, n_heads, seq_len, d_head] x [batch_size, n_heads, d_head, seq_len]
    grad_attn_probs = torch.matmul(grad_attn_output, v.transpose(-2, -1))

    # Step 4: Gradient w.r.t. scores (before softmax)
    # Softmax gradient: dL/ds = P * (dL/dP - sum(dL/dP * P))
    # where P is attention probabilities and s is scores
    sum_term = torch.sum(grad_attn_probs * attn_probs, dim=-1, keepdim=True)
    grad_scores = attn_probs * (grad_attn_probs - sum_term)
    grad_scores[mask == 0] = 0.0

    # Step 5: Gradient w.r.t. Q
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_Q = torch.matmul(grad_scores, k) * softmax_scale
    if dq is None:
        dq = grad_Q
    else:
        dq.copy_(grad_Q)

    # Step 6: Gradient w.r.t. K
    # [batch_size, n_heads, seq_len, seq_len] x [batch_size, n_heads, seq_len, d_head]
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), q) * softmax_scale
    if dk is None:
        dk = grad_K
    else:
        dk.copy_(grad_K)

    return dq, dk, dv
