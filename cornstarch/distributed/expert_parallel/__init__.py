"""Expert parallelism for MoE models, built around batched experts.

Batched experts are the canonical representation: all experts of a layer share
a few stacked weight tensors (a leading expert dimension) and run through one
grouped computation.  This is how modern Hugging Face MoE models store experts
(Qwen3.5, DeepSeek, Llama4, GPT-OSS, Mixtral all use a single ``*Experts``
module via ``@use_experts_implementation``), and it is what expert parallelism
shards.

``apply_expert_parallel`` therefore does, for every MoE layer:

1. **If the layer already uses batched experts** — parallelize them: slice the
   stacked weight tensors along the expert dimension so each EP rank keeps only
   its ``num_experts // ep_size`` experts, and wrap the experts' forward with
   ``ExpertParallelDispatcher`` (all-to-all token routing).  The module's own
   per-expert math is reused on the sharded weights, so any batched layout is
   supported.
2. **If the layer stores experts as an** ``nn.ModuleList`` — convert it to a
   batched experts module first (stack the per-expert weights, rewrite the
   block forward to the batched convention), then parallelize it through the
   same path above.

``ep_size`` only has to divide ``num_experts``; a rank may own several experts.
Sharded expert parameters are tagged ``_is_expert_parallel = True`` so callers
can distinguish EP-axis collectives (which must skip them) from orthogonal
DP/CP collectives (which synchronize corresponding local expert shards).

``apply_expert_parallel`` slices real tensors, so call it after
``module.materialize()``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from cornstarch.distributed.expert_parallel.routing import (
    ExpertParallelDispatcher,
    ExpertRouter,
)
from cornstarch.models.model_base import CornstarchModelBase


# ---------------------------------------------------------------------------
# Batched experts: detection, the canonical module, and parallelization
# ---------------------------------------------------------------------------

class _BatchedExperts(nn.Module):
    """Experts stored as stacked weight tensors (the canonical EP form).

    Used as the conversion target for ``nn.ModuleList`` experts.  Weights are
    ``gate_up_proj`` ``(num_experts, 2 * intermediate, hidden)`` (the gate and
    up projections concatenated) and ``down_proj``
    ``(num_experts, hidden, intermediate)``.  The forward mirrors the
    Hugging Face batched-expert convention: it takes the per-token top-k expert
    indices and weights and returns the weighted sum over each token's experts.
    """

    def __init__(
        self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn: nn.Module
    ) -> None:
        super().__init__()
        self.num_experts = gate_up_proj.shape[0]
        self.gate_up_proj = nn.Parameter(gate_up_proj)
        self.down_proj = nn.Parameter(down_proj)
        self.act_fn = act_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current = hidden_states[token_idx]
            gate, up = F.linear(current, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_idx])
            current = current * top_k_weights[token_idx, top_k_pos, None]
            final = final.index_add(0, token_idx, current.to(final.dtype))
        return final


def _is_batched_experts(module: nn.Module) -> bool:
    """Return whether ``module`` stores experts as stacked weight tensors.

    Identifies an experts module by an integer ``num_experts`` attribute plus at
    least two of its own parameters whose leading dimension is ``num_experts``
    (e.g. ``gate_up_proj`` and ``down_proj``).  The "at least two" rule keeps
    single-weight routers (which also expose ``num_experts``) from matching.
    """
    num_experts = getattr(module, "num_experts", None)
    if not isinstance(num_experts, int) or num_experts <= 0:
        return False
    stacked = sum(
        1
        for p in module.parameters(recurse=False)
        if p.dim() >= 2 and p.shape[0] == num_experts
    )
    return stacked >= 2


def _find_batched_experts(layer: nn.Module) -> list[nn.Module]:
    """Return batched-expert modules nested anywhere inside a decoder layer."""
    return [m for m in layer.modules() if _is_batched_experts(m)]


def _parallelize_experts(
    experts: nn.Module,
    ep_rank: int,
    ep_size: int,
    ep_group: dist.ProcessGroup,
) -> None:
    """Shard a batched experts module across EP ranks and inject all-to-all.

    Every parameter whose leading dimension is the global expert count is sliced
    to this rank's contiguous expert range, ``num_experts`` is updated to the
    local count, and the module's forward is wrapped: tokens are dispatched to
    the rank that owns each chosen expert, the module's *own* per-expert
    computation runs on the received tokens (so any batched layout is handled),
    and the weighted results are collected back.
    """
    num_experts = int(experts.num_experts)
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})."
        )
    experts_per_rank = num_experts // ep_size
    start = ep_rank * experts_per_rank
    end = start + experts_per_rank

    for name, param in list(experts.named_parameters(recurse=False)):
        if param.dim() >= 1 and param.shape[0] == num_experts:
            sharded = nn.Parameter(
                param.data[start:end].clone(), requires_grad=param.requires_grad
            )
            sharded._is_expert_parallel = True
            setattr(experts, name, sharded)
    experts.num_experts = experts_per_rank

    # The module's pre-wrap forward already knows how to compute its own layout;
    # run it on the locally-owned experts only.  Each received token carries a
    # single (rank-local) expert index, and routing weights are applied later by
    # ``collect``, so we pass unit weights here.
    base_forward = experts.forward
    dispatcher = ExpertParallelDispatcher()

    def ep_forward(
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        local_tokens, local_ids, recv_counts = dispatcher.dispatch(
            hidden_states, top_k_index, ep_group, num_experts
        )
        unit_weights = torch.ones(
            (local_tokens.shape[0], 1),
            dtype=top_k_weights.dtype,
            device=local_tokens.device,
        )
        local_out = base_forward(local_tokens, local_ids.unsqueeze(1), unit_weights)
        return dispatcher.collect(
            local_out, top_k_weights, top_k_index, recv_counts, ep_group
        )

    experts.forward = ep_forward


# ---------------------------------------------------------------------------
# ModuleList experts: convert to the batched representation
# ---------------------------------------------------------------------------

_GATE_PROJ_NAMES = ("gate_proj", "w1")
_UP_PROJ_NAMES = ("up_proj", "w3")
_DOWN_PROJ_NAMES = ("down_proj", "w2")


def _find_modulelist_moe_blocks(layer: nn.Module) -> list[nn.Module]:
    """Return MoE blocks that store experts as an ``nn.ModuleList`` + linear gate."""
    blocks = []
    for module in layer.modules():
        experts = getattr(module, "experts", None)
        gate = getattr(module, "gate", None)
        if isinstance(experts, nn.ModuleList) and isinstance(gate, nn.Linear):
            blocks.append(module)
    return blocks


def _expert_linear(expert: nn.Module, names: tuple[str, ...]) -> nn.Linear:
    for name in names:
        candidate = getattr(expert, name, None)
        if isinstance(candidate, nn.Linear):
            return candidate
    raise NotImplementedError(
        f"Cannot convert expert of type {type(expert).__name__} to batched form: "
        f"expected one of {names} to be an nn.Linear."
    )


def _convert_modulelist_block_to_batched(block: nn.Module) -> None:
    """Convert an ``nn.ModuleList`` MoE block in place to use batched experts.

    Stacks each expert's gate/up/down projection weights into the canonical
    ``gate_up_proj`` / ``down_proj`` tensors, replaces ``block.experts`` with a
    :class:`_BatchedExperts` module, and rewrites ``block.forward`` to the
    batched convention (route with the existing linear gate, then call the
    batched experts).  This targets the common layout where the block returns
    the MoE hidden states; blocks with extra outputs (e.g. router logits) or a
    shared expert need a model-specific adapter.
    """
    experts: nn.ModuleList = block.experts
    gate: nn.Linear = block.gate
    num_experts = len(experts)
    top_k = int(getattr(block, "top_k", getattr(block, "num_experts_per_tok", 2)))

    gate_up_rows, down_rows = [], []
    for expert in experts:
        gate_proj = _expert_linear(expert, _GATE_PROJ_NAMES)
        up_proj = _expert_linear(expert, _UP_PROJ_NAMES)
        down_proj = _expert_linear(expert, _DOWN_PROJ_NAMES)
        gate_up_rows.append(torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0))
        down_rows.append(down_proj.weight.data)

    act_fn = getattr(experts[0], "act_fn", None) or nn.SiLU()
    batched = _BatchedExperts(
        torch.stack(gate_up_rows, dim=0), torch.stack(down_rows, dim=0), act_fn
    )
    block.experts = batched

    router = ExpertRouter(num_experts, top_k)

    def block_forward(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        routing_weights, expert_ids = router(gate(flat))
        out = block.experts(flat, expert_ids, routing_weights)
        return out.reshape(shape)

    block.forward = block_forward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_expert_parallel(
    module: CornstarchModelBase,
    ep_group: dist.ProcessGroup,
) -> None:
    """Shard and EP-patch every MoE layer in the module.

    For each decoder layer, ``nn.ModuleList`` MoE blocks are first converted to
    batched experts, then every batched experts module is sharded across the EP
    ranks and wrapped with all-to-all dispatch.

    Call after ``module.materialize()``: the expert weights are sliced in place,
    so they must already hold real tensors.
    """
    ep_rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)

    _, layers_name, _ = module._section_names()
    for layer in getattr(module, layers_name):
        for block in _find_modulelist_moe_blocks(layer):
            _convert_modulelist_block_to_batched(block)
        for experts in _find_batched_experts(layer):
            _parallelize_experts(experts, ep_rank, ep_size, ep_group)
