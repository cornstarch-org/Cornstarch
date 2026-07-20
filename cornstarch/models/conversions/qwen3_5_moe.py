from __future__ import annotations

import copy
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from cornstarch.models.conversions.qwen3_5 import Qwen3_5LanguageForwardSpec
from cornstarch.models.forward_specs import LayerContext
from cornstarch.models.language_model import CornstarchLanguageModel


class _ForwardSumIdentityBackward(torch.autograd.Function):
    """All-reduce values without duplicating gradients in the backward pass."""

    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        output = tensor.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def _reduce_router_statistics(
    statistics: torch.Tensor,
    *,
    cp_group: dist.ProcessGroup | None,
    dp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Reduce sufficient router statistics with axis-correct autograd."""
    if cp_group is not None and dist.get_world_size(cp_group) > 1:
        # CP parameters later receive a SUM gradient synchronization. Only the
        # value reduction belongs here; an autograd reduction would count every
        # token once per CP rank.
        statistics = _ForwardSumIdentityBackward.apply(statistics, cp_group)
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        # DP gradients are averaged later, so the differentiable SUM's backward
        # replication is intentionally cancelled by that AVG.
        statistics = dist_nn.all_reduce(
            statistics, op=dist.ReduceOp.SUM, group=dp_group
        )
    return statistics


class QwenMoeLanguageForwardSpec(Qwen3_5LanguageForwardSpec):
    """Native forward spec for Qwen3.5 MoE text models."""

    output_cls = MoeCausalLMOutputWithPast

    def prepare_layer_context(
        self, model: nn.Module, hidden_states: torch.Tensor, **kwargs: Any
    ) -> LayerContext:
        context = super().prepare_layer_context(model, hidden_states, **kwargs)
        context["router_aux_statistics"] = kwargs.get("router_aux_statistics")
        context["router_logits_tensor"] = kwargs.get("router_logits_tensor")
        context["router_attention_mask"] = kwargs.get("attention_mask")
        return context

    def process_layer_output(
        self,
        model: nn.Module,
        layer_idx: int,
        layer_output: Any,
        context: LayerContext,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = super().process_layer_output(
            model, layer_idx, layer_output, context, **kwargs
        )
        local_index = layer_idx - int(getattr(model, "_pipeline_layer_offset", 0))
        if not (0 <= local_index < len(model.decoder_layers)):
            return hidden_states
        gate = getattr(getattr(model.decoder_layers[local_index], "mlp", None), "gate", None)
        router_logits = getattr(gate, "_cornstarch_router_logits", None)
        if router_logits is None:
            return hidden_states
        probabilities = F.softmax(router_logits, dim=-1)
        selected = torch.topk(
            probabilities, model.config.num_experts_per_tok, dim=-1
        ).indices
        expert_mask = F.one_hot(
            selected, num_classes=model.config.num_experts
        ).float()
        token_mask = context.get("router_attention_mask")
        if token_mask is None:
            valid = torch.ones(
                router_logits.shape[0],
                device=router_logits.device,
                dtype=probabilities.dtype,
            )
        else:
            valid = token_mask.reshape(-1).to(
                device=router_logits.device, dtype=probabilities.dtype
            )
        counts = (expert_mask * valid[:, None, None]).sum(dim=0)
        probability_sums = (probabilities * valid[:, None]).sum(dim=0)
        statistics = torch.cat(
            [counts.reshape(-1), probability_sums, valid.sum().reshape(1)]
        )
        previous = context.get("router_aux_statistics")
        context["router_aux_statistics"] = (
            statistics if previous is None else previous + statistics
        )
        previous_logits = context.get("router_logits_tensor")
        context["router_logits_tensor"] = (
            router_logits
            if previous_logits is None
            else torch.cat([previous_logits, router_logits], dim=0)
        )
        return hidden_states

    def build_output(
        self, model: nn.Module, hidden_states: torch.Tensor, context: LayerContext, **kwargs: Any
    ) -> MoeCausalLMOutputWithPast:
        output = super().build_output(model, hidden_states, context, **kwargs)
        output_router_logits = kwargs.get(
            "output_router_logits", model.config.output_router_logits
        )
        statistics = (
            context.get("router_aux_statistics") if output_router_logits else None
        )
        aux_loss = None
        if statistics is not None:
            statistics = _reduce_router_statistics(
                statistics,
                cp_group=getattr(model, "_cp_group", None),
                dp_group=getattr(model, "_dp_group", None),
            )
            num_experts = model.config.num_experts
            top_k = model.config.num_experts_per_tok
            count_size = top_k * num_experts
            counts = statistics[:count_size].reshape(top_k, num_experts)
            probability_sums = statistics[count_size : count_size + num_experts]
            token_count = statistics[-1].clamp_min(1)
            aux_loss = num_experts * torch.sum(
                (counts / token_count) * (probability_sums / token_count)[None]
            )
        loss = output.loss
        if loss is not None and aux_loss is not None:
            loss = loss + model.config.router_aux_loss_coef * aux_loss.to(loss)
        router_logits = None
        logits_tensor = context.get("router_logits_tensor")
        if output_router_logits and logits_tensor is not None:
            router_logits = tuple(
                logits_tensor.chunk(model.config.num_hidden_layers, dim=0)
            )
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=output.logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            router_logits=router_logits,
        )


def _capture_router_logits(layer: nn.Module) -> None:
    """Retain the router output that the reused HF decoder intentionally drops."""
    gate = getattr(getattr(layer, "mlp", None), "gate", None)
    if gate is None:
        return
    base_forward = gate.forward

    def forward(this: nn.Module, *args: Any, **kwargs: Any) -> Any:
        output = base_forward(*args, **kwargs)
        this._cornstarch_router_logits = output[0]
        return output

    gate.forward = MethodType(forward, gate)


def convert_qwen3_5_moe_config(
    config: PretrainedConfig,
    attn_implementation: str | None = None,
    layer_offload_config=None,
    layer_compile_config=None,
) -> CornstarchLanguageModel:
    """Convert a Qwen3.5 MoE text config into a Cornstarch language model."""
    with torch.device("meta"):
        hf_model = Qwen3_5MoeForCausalLM(copy.deepcopy(config))
    for layer in hf_model.model.layers:
        _capture_router_logits(layer)
    return CornstarchLanguageModel(
        config,
        pre_decoder={
            "embed_tokens": hf_model.model.embed_tokens,
            "rotary_emb": hf_model.model.rotary_emb,
        },
        decoder_layers=hf_model.model.layers,
        post_decoder={"norm": hf_model.model.norm, "lm_head": hf_model.lm_head},
        hf_to_cornstarch_prefixes=(
            ("model.embed_tokens.", "pre_decoder.embed_tokens."),
            ("model.rotary_emb.", "pre_decoder.rotary_emb."),
            ("model.layers.", "decoder_layers."),
            ("model.norm.", "post_decoder.norm."),
            ("lm_head.", "post_decoder.lm_head."),
        ),
        hf_model_factory=Qwen3_5MoeForCausalLM,
        forward_spec=QwenMoeLanguageForwardSpec(),
        attn_implementation=attn_implementation,
        layer_offload_config=layer_offload_config,
        layer_compile_config=layer_compile_config,
    )
