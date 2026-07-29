"""Declarative DTensor plans for converted Cornstarch layer structures.

Each dict maps submodule paths *within a single repeated layer* to a
``ParallelStyle`` that ``parallelize_module`` applies.  The caller iterates
over the model's repeated layers and passes each layer plus this dict.

Only converted, tested structures belong here. Keeping speculative plans for
families that ``from_hf_config`` cannot construct creates a false promise: the
plan is unreachable and cannot be validated through Cornstarch's unified model
lifecycle. Hybrid models select a plan by semantic ``layer_type``; the TP
implementation itself does not contain a family-specific branch ladder.
"""

from __future__ import annotations

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


DENSE_ATTN_MLP_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
}

ATTENTION_ONLY_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.o_proj": RowwiseParallel(),
}

ENCODER_ATTN_MLP_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.out_proj": RowwiseParallel(),
    "mlp.fc1": ColwiseParallel(),
    "mlp.fc2": RowwiseParallel(),
}

GEMMA4_VISION_TP_PLAN: dict = {
    "self_attn.q_proj.linear": ColwiseParallel(),
    "self_attn.k_proj.linear": ColwiseParallel(),
    "self_attn.v_proj.linear": ColwiseParallel(),
    "self_attn.o_proj.linear": RowwiseParallel(),
    "mlp.gate_proj.linear": ColwiseParallel(),
    "mlp.up_proj.linear": ColwiseParallel(),
    "mlp.down_proj.linear": RowwiseParallel(),
}
WHISPER_TP_PLAN: dict = {
    "self_attn.q_proj": ColwiseParallel(),
    "self_attn.k_proj": ColwiseParallel(),
    "self_attn.v_proj": ColwiseParallel(),
    "self_attn.out_proj": RowwiseParallel(),
    "fc1": ColwiseParallel(),
    "fc2": RowwiseParallel(),
}
# MoE layers shard the token mixer over TP while their router and expert FFNs
# remain replicated across TP lanes; EP independently owns expert placement.
GATED_DELTA_TP_PLAN: dict = {
    "linear_attn.in_proj_z": ColwiseParallel(),
    "linear_attn.in_proj_a": ColwiseParallel(),
    "linear_attn.in_proj_b": ColwiseParallel(),
    "linear_attn.out_proj": RowwiseParallel(),
}

DENSE_GATED_DELTA_TP_PLAN: dict = {
    **GATED_DELTA_TP_PLAN,
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
}
_DEFAULT_PLANS: dict[str, dict] = {
    "LlamaConfig": DENSE_ATTN_MLP_TP_PLAN,
    "Qwen3_5MoeTextConfig": ATTENTION_ONLY_TP_PLAN,
    "CLIPVisionConfig": ENCODER_ATTN_MLP_TP_PLAN,
    "Gemma4VisionConfig": GEMMA4_VISION_TP_PLAN,
    "Llama4TextConfig": ATTENTION_ONLY_TP_PLAN,
    "Siglip2VisionConfig": ENCODER_ATTN_MLP_TP_PLAN,
    "WhisperConfig": WHISPER_TP_PLAN,
    "Qwen3_5TextConfig": DENSE_ATTN_MLP_TP_PLAN,
}

_LAYER_TYPE_PLANS: dict[str, dict[str, dict]] = {
    "Qwen3_5TextConfig": {
        "full_attention": DENSE_ATTN_MLP_TP_PLAN,
        "linear_attention": DENSE_GATED_DELTA_TP_PLAN,
    },
    # MoE experts and routers remain replicated over TP lanes and are sharded
    # only over EP, so TP owns the token mixer and nothing else.
    "Qwen3_5MoeTextConfig": {
        "full_attention": ATTENTION_ONLY_TP_PLAN,
        "linear_attention": GATED_DELTA_TP_PLAN,
    },
}


def get_tp_plan(config_class_name: str) -> dict | None:
    """Return the per-layer TP plan for ``config_class_name``, or ``None`` if unregistered."""
    return _DEFAULT_PLANS.get(config_class_name)


def get_layer_tp_plan(config_class_name: str, layer_type: str | None) -> dict | None:
    """Resolve a hybrid layer by semantic type, rejecting partial coverage."""
    layer_plans = _LAYER_TYPE_PLANS.get(config_class_name)
    if layer_plans is not None:
        try:
            return layer_plans[layer_type]
        except KeyError as error:
            raise ValueError(
                f"Unsupported layer type {layer_type!r} for "
                f"{config_class_name}; refusing partial TP."
            ) from error
    return get_tp_plan(config_class_name)
