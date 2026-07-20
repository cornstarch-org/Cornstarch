"""Context parallelism injection.

``apply_context_parallel`` registers the CP all-gather flash attention in HF's
``ALL_ATTENTION_FUNCTIONS`` registry and sets every reused Hugging Face leaf
config in the module to dispatch to it.

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

_CP_ATTN_KEY_PREFIX = "context_parallel"


def apply_context_parallel(
    module: CornstarchModelBase,
    cp_group: dist.ProcessGroup,
    causal: bool = False,
) -> str:
    """Inject CP all-gather flash attention into the module.

    Registers a group-bound CP attention function under a module-specific key
    and sets both the Cornstarch wrapper config and every reused Hugging Face
    leaf config to that key. Reused attention modules own config objects that
    are distinct from ``module.hf_config``; updating only the wrapper silently
    leaves real forwards on their previous attention implementation. A unique
    key is also required because different modalities can own different CP
    process groups in the same process.

    ``causal=True`` binds the kernel to the causal per-run prefix+diagonal
    decomposition; the per-rank global positions are recovered at call time by
    all-gathering the ``position_ids`` HF forwards into the attention callable
    (no splitter instance is threaded through).  The default (``causal=False``)
    is full non-causal attention, unchanged from before.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    attention_key = f"{_CP_ATTN_KEY_PREFIX}_{id(module):x}"
    ALL_ATTENTION_FUNCTIONS.register(
        attention_key,
        functools.partial(
            context_parallel_flash_attention, cp_group=cp_group, causal=causal
        ),
    )

    configs = [module.hf_config]
    configs.extend(
        config
        for child in module.modules()
        if (config := getattr(child, "config", None)) is not None
        and hasattr(config, "_attn_implementation")
    )
    seen: set[int] = set()
    for config in configs:
        if id(config) in seen:
            continue
        seen.add(id(config))
        config._attn_implementation = attention_key
    return attention_key
