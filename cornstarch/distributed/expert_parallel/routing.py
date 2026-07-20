"""Expert routing and all-to-all dispatch for expert parallelism.

``ExpertRouter`` computes top-k softmax routing weights and expert IDs.
``ExpertParallelDispatcher`` performs two all-to-all exchanges: one to send
tokens to the EP rank that owns each assigned expert, and one to collect
the processed results back.  Both classes are pure PyTorch and model-family
agnostic; they are injected into MoE layers by ``apply_expert_parallel()``.

Each EP rank owns ``num_experts // ep_size`` *consecutive* experts, so
``ep_size`` only has to divide ``num_experts`` — a rank can hold several
experts.  The token exchange goes through :class:`_AllToAllSingle`, an
autograd-aware wrapper around ``dist.all_to_all_single`` whose backward is
the transposed exchange (send/recv split sizes swapped); this lets gradients
flow back through both dispatch and collect so experts train normally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class _AllToAllSingle(torch.autograd.Function):
    """Autograd-aware ``dist.all_to_all_single`` with uneven splits.

    The backward pass is itself an all-to-all with the input/output split
    sizes swapped, which is the exact adjoint of the forward exchange.
    """

    @staticmethod
    def forward(ctx, input_, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        input_ = input_.contiguous()
        output = input_.new_empty((sum(output_split_sizes), *input_.shape[1:]))
        dist.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_empty(
            (sum(ctx.input_split_sizes), *grad_output.shape[1:])
        )
        dist.all_to_all_single(
            grad_input,
            grad_output,
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_input, None, None, None


def _all_to_all_single(
    input_: torch.Tensor,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Differentiable variable-split all-to-all of ``input_``'s rows."""
    return _AllToAllSingle.apply(input_, output_split_sizes, input_split_sizes, group)


class ExpertRouter(nn.Module):
    """Top-k softmax router for MoE layers.

    Selects the ``top_k`` experts with the highest gate logits per token and
    returns softmax-normalized routing weights for the weighted combination
    of expert outputs.
    """

    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self, gate_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(routing_weights, expert_ids)`` both of shape ``(N, top_k)``."""
        routing_weights, expert_ids = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        return routing_weights, expert_ids


class ExpertParallelDispatcher:
    """All-to-all token dispatch and collect for expert parallelism.

    Each EP rank owns ``num_experts // ep_size`` consecutive experts.
    ``dispatch`` sorts tokens by destination rank and sends them via
    all-to-all; ``collect`` reverses the exchange and applies routing weights
    to produce the final weighted output.

    The dispatcher stores the sort permutation from ``dispatch`` so that
    ``collect`` can restore the original token ordering.  The caller runs
    local experts between the two calls.
    """

    def __init__(self) -> None:
        self._unsort_idx: torch.Tensor | None = None

    def dispatch(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        ep_group: dist.ProcessGroup,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Send each token to the EP rank that owns its assigned expert.

        ``num_experts`` is the global expert count; with ``ep_size`` ranks each
        rank owns ``num_experts // ep_size`` consecutive experts.  Returns
        ``(local_tokens, local_expert_ids, recv_counts)`` where ``local_tokens``
        are the tokens this rank received, ``local_expert_ids`` are the
        rank-local expert indices (in ``[0, experts_per_rank)``), and
        ``recv_counts`` records how many tokens came from each source rank
        (needed by ``collect``).
        """
        ep_size = dist.get_world_size(ep_group)
        if num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"ep_size ({ep_size})."
            )
        _, d = tokens.shape
        top_k = expert_ids.shape[1]
        experts_per_rank = num_experts // ep_size

        flat_expert_ids = expert_ids.reshape(-1)
        flat_tokens = tokens.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, d)

        dest_ranks = flat_expert_ids // experts_per_rank

        sort_idx = torch.argsort(dest_ranks, stable=True)
        sorted_tokens = flat_tokens[sort_idx]
        sorted_expert_ids = flat_expert_ids[sort_idx] % experts_per_rank

        unsort_idx = torch.empty_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(len(sort_idx), device=sort_idx.device)
        self._unsort_idx = unsort_idx

        send_counts = torch.zeros(ep_size, dtype=torch.long, device=tokens.device)
        for r in range(ep_size):
            send_counts[r] = (dest_ranks == r).sum()

        recv_counts = torch.zeros(ep_size, dtype=torch.long, device=tokens.device)
        dist.all_to_all(
            list(recv_counts.split(1)),
            list(send_counts.split(1)),
            group=ep_group,
        )

        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()

        # Token payload goes through the differentiable exchange so gradients
        # reach the dispatched hidden states during backward.
        recv_tokens = _all_to_all_single(sorted_tokens, recv_list, send_list, ep_group)

        # Expert ids are integer metadata — no gradient needed.
        recv_expert_ids = torch.zeros(
            sum(recv_list), dtype=torch.long, device=tokens.device
        )
        dist.all_to_all_single(
            recv_expert_ids,
            sorted_expert_ids,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=ep_group,
        )

        return recv_tokens, recv_expert_ids, recv_counts

    def collect(
        self,
        expert_outputs: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_ids: torch.Tensor,
        recv_counts: torch.Tensor,
        ep_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Collect processed tokens back and apply routing weights.

        Reverses the all-to-all from ``dispatch``, restores the original token
        ordering, and returns the weighted sum across each token's top-k experts
        as a tensor of shape ``(N, d)``.
        """
        ep_size = dist.get_world_size(ep_group)
        N, d = routing_weights.shape[0], expert_outputs.shape[-1]

        send_counts = recv_counts
        recv_count_list = [
            torch.zeros(1, dtype=torch.long, device=expert_outputs.device)
            for _ in range(ep_size)
        ]
        dist.all_to_all(
            recv_count_list,
            list(send_counts.split(1)),
            group=ep_group,
        )

        out_split = [t.item() for t in recv_count_list]
        in_split = recv_counts.tolist()
        combined = _all_to_all_single(expert_outputs, out_split, in_split, ep_group)

        if self._unsort_idx is not None:
            combined = combined[self._unsort_idx]

        top_k = expert_ids.shape[1]
        combined = combined.view(N, top_k, d)
        weights = routing_weights.unsqueeze(-1)
        return (combined * weights).sum(dim=1)
