"""Per-model-family DTensor TP plan dicts.

Each dict maps submodule paths *within a single repeated layer* to a
``ParallelStyle`` that ``parallelize_module`` applies.  The caller iterates
over the model's repeated layers and passes each layer plus this dict.

Naming follows the HF config class name (e.g. ``LlamaConfig``) so the
lookup stays dependency-free.
"""
from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


# ---------------------------------------------------------------------------
# Llama / Llama 2 / Llama 3
# ---------------------------------------------------------------------------
LLAMA_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
}

# Mistral / Qwen2 / Gemma share the same projection names as Llama.
MISTRAL_TP_PLAN: dict = LLAMA_TP_PLAN
QWEN2_TP_PLAN: dict = LLAMA_TP_PLAN
GEMMA_TP_PLAN: dict = LLAMA_TP_PLAN

# ---------------------------------------------------------------------------
# Mixtral (MoE: gate stays replicated; expert FFN uses EP — TP only on attn)
# ---------------------------------------------------------------------------
MIXTRAL_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
}

# ---------------------------------------------------------------------------
# Phi-3 (fused qkv_proj + gate_up_proj)
# ---------------------------------------------------------------------------
PHI3_TP_PLAN: dict = {
    "self_attn.qkv_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
    "mlp.gate_up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
}

# ---------------------------------------------------------------------------
# GPT-2
# ---------------------------------------------------------------------------
GPT2_TP_PLAN: dict = {
    "attn.c_attn": ColwiseParallel(),
    "attn.c_proj": RowwiseParallel(),
    "mlp.c_fc": ColwiseParallel(),
    "mlp.c_proj": RowwiseParallel(),
}

# ---------------------------------------------------------------------------
# BERT / RoBERTa
# ---------------------------------------------------------------------------
BERT_TP_PLAN: dict = {
    "attention.self.query": ColwiseParallel(),
    "attention.self.key": ColwiseParallel(),
    "attention.self.value": ColwiseParallel(),
    "attention.output.dense": RowwiseParallel(),
    "intermediate.dense": ColwiseParallel(),
    "output.dense": RowwiseParallel(),
}

# ---------------------------------------------------------------------------
# Lookup: HF config class name → per-layer TP plan
# ---------------------------------------------------------------------------
# MoE families: shard attention only (expert FFNs use EP, the router/shared
# expert stay replicated), exactly like the Mixtral plan.  This lets tensor
# parallelism compose with expert parallelism on the same model.
QWEN_MOE_TP_PLAN: dict = MIXTRAL_TP_PLAN

_REGISTRY: dict[str, dict] = {
    "LlamaConfig": LLAMA_TP_PLAN,
    "Llama4Config": LLAMA_TP_PLAN,
    "Qwen3_5MoeTextConfig": QWEN_MOE_TP_PLAN,
    "MistralConfig": MISTRAL_TP_PLAN,
    "Qwen2Config": QWEN2_TP_PLAN,
    "Qwen2_5Config": QWEN2_TP_PLAN,
    "GemmaConfig": GEMMA_TP_PLAN,
    "Gemma2Config": GEMMA_TP_PLAN,
    "Gemma3Config": GEMMA_TP_PLAN,
    "MixtralConfig": MIXTRAL_TP_PLAN,
    "Phi3Config": PHI3_TP_PLAN,
    "GPT2Config": GPT2_TP_PLAN,
    "BertConfig": BERT_TP_PLAN,
    "RobertaConfig": BERT_TP_PLAN,
}


def get_tp_plan(config_class_name: str) -> dict | None:
    """Return the per-layer TP plan for ``config_class_name``, or ``None`` if unregistered."""
    return _REGISTRY.get(config_class_name)
