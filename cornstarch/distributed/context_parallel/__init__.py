"""Context parallelism injection.

``apply_context_parallel`` registers the CP all-gather flash attention in HF's
``ALL_ATTENTION_FUNCTIONS`` registry and sets the module's config to dispatch to
it via ``config._attn_implementation = "context_parallel"``.

Sequence splitting (CP token assignment) is intentionally kept out of the
model.  Callers use a :class:`~cornstarch.distributed.context_parallel.splitters.ContextParallelSplitter`
in their dataloader / collation step to slice inputs before the model call.
"""
from __future__ import annotations

import functools

import torch.distributed as dist

from cornstarch.distributed.context_parallel.attention import (
    context_parallel_flash_attention,
)
from cornstarch.models.model_base import CornstarchModelBase

_CP_ATTN_KEY = "context_parallel"


def apply_context_parallel(
    module: CornstarchModelBase,
    cp_group: dist.ProcessGroup,
    causal: bool = False,
) -> None:
    """Inject CP all-gather flash attention into the module.

    Registers a group-bound CP attention function in HF's
    ``ALL_ATTENTION_FUNCTIONS["context_parallel"]`` and sets the module's
    ``hf_config._attn_implementation`` so all attention layers dispatch to
    the CP kernel.  Works on both meta and materialized modules.

    ``causal=True`` binds the kernel to the causal per-run prefix+diagonal
    decomposition; the per-rank global positions are recovered at call time by
    all-gathering the ``position_ids`` HF forwards into the attention callable
    (no splitter instance is threaded through).  The default (``causal=False``)
    is full non-causal attention, unchanged from before.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[_CP_ATTN_KEY] = functools.partial(
        context_parallel_flash_attention, cp_group=cp_group, causal=causal
    )
    module.hf_config._attn_implementation = _CP_ATTN_KEY
