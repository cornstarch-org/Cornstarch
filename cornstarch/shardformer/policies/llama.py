from __future__ import annotations

import functools
import itertools
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    PaddingEmbedding,
    PaddingLMHead,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    SubModuleReplacementDescription,
)
from colossalai.shardformer.policies.llama import (
    LlamaForCausalLMPolicy as ColossalLlamaForCausalLMPolicy,
)
from colossalai.shardformer.policies.llama import (
    LlamaForSequenceClassificationPolicy as ColossalLlamaForSequenceClassificationPolicy,
)
from colossalai.shardformer.policies.llama import (
    LlamaModelPolicy as ColossalLlamaModelPolicy,
)
from colossalai.shardformer.policies.llama import (
    LlamaPolicy as ColossalLlamaPolicy,
)
from torch import Tensor, nn
from transformers import LlamaConfig, PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal
from transformers.models.llama.modeling_llama import LlamaModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.llama import (
    LlamaAttentionForwards,
    LlamaModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

__all__ = [
    "LlamaPolicy",
    "LlamaForCausalLMPolicy",
    "LlamaForSequenceClassificationPolicy",
]


class LlamaPolicy(PipelineTemplatePolicyBase, ColossalLlamaPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, LlamaConfig
        ), "config must be an instance of LlamaConfig"
        config: LlamaConfig = cast(LlamaConfig, config)

        modules = []
        modules.append("embed_tokens")
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.llama.modeling_llama" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "LlamaModel" else "model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embed_tokens" not in modules_in_template[0]:
            raise ValueError("embed_tokens must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaDecoderLayer,
            LlamaFlashAttention2,
            LlamaSdpaAttention,
        )

        config: LlamaConfig = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": LlamaAttention,
            "flash_attention_2": LlamaFlashAttention2,
            "sdpa": LlamaSdpaAttention,
        }
        attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]

        policy = {}

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        if sp_mode == "ring_attn" and not self.is_causal:
            raise ValueError(
                "Ring attention is only meant for causal language modeling."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_q_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        hidden_size = config.hidden_size

        if sp_mode == "all_to_all":
            # Ulysses all-to-all context parallelism needs to partition number of heads
            hidden_size //= sp_size

            assert (
                num_q_heads % sp_size == 0
            ), "The number of attention heads must be divisible by the sequence parallel size."
            num_q_heads //= sp_size

            if num_kv_heads:
                assert (
                    num_kv_heads % sp_size == 0
                ), "The number of key_value heads must be divisible by the sequence parallel size."
                num_kv_heads //= sp_size

        if self.shard_config.enable_tensor_parallelism:
            hidden_size //= tp_size

            assert (
                num_q_heads % tp_size == 0
            ), "The number of attention heads must be divisible by the tensor parallel size."
            num_q_heads //= tp_size

            if num_kv_heads:
                assert (
                    num_kv_heads % tp_size == 0
                ), "The number of key_value heads must be divisible by the tensor parallel size."
                num_kv_heads //= tp_size

        attention_attribute_replacement = {}
        attention_attribute_replacement["hidden_size"] = hidden_size
        attention_attribute_replacement["num_heads"] = num_q_heads
        if num_kv_heads:
            attention_attribute_replacement["num_key_value_heads"] = num_kv_heads

        policy[attn_cls] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": functools.partial(
                    LlamaAttentionForwards.forward,
                    shard_config=self.shard_config,
                )
            },
        )

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

            policy[LlamaModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                ],
            )

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        elif self.tie_weight:
            embedding_cls = PaddingEmbedding

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
                    },
                ),
                policy=policy,
                target_key=LlamaModel,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=LlamaDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=LlamaModel,
            )

        return policy


class LlamaModelPolicy(LlamaPolicy, ColossalLlamaModelPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return LlamaPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    LlamaModelForwards.llama_model_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=LlamaModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        layers = super().get_held_layers()
        layers.append(self.model.rotary_emb)

        return layers


class LlamaForCausalLMPolicy(LlamaPolicy, ColossalLlamaForCausalLMPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in LlamaPolicy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM

        self.is_causal = True
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            target_module = VocabParallelLMHead1D
            kwargs = {
                "gather_output": not self.shard_config.parallel_output,
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
            }
        else:
            target_module = PaddingLMHead
            kwargs = {
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
            }

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="lm_head",
                target_module=target_module,
                kwargs=kwargs,
            ),
            policy=policy,
            target_key=LlamaForCausalLM,
        )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    LlamaModelForwards.llama_for_causal_lm_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=LlamaForCausalLM,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            layers.append(self.model.lm_head)
        layers.append(self.model.model.rotary_emb)

        return layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        llama_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(llama_model.embed_tokens.weight) == id(self.model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: llama_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.lm_head.weight,
                    }
                ]

        return []


class LlamaForSequenceClassificationPolicy(
    LlamaPolicy, ColossalLlamaForSequenceClassificationPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in LlamaPolicy.get_all_modules(config)]
        modules.append("score")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "score" not in template.modules_per_stage[-1]:
            raise ValueError("score must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.llama.modeling_llama import (
            LlamaForSequenceClassification,
        )

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="score",
                    target_module=Linear1D_Col,
                    kwargs=dict(gather_output=True),
                ),
                policy=policy,
                target_key=LlamaForSequenceClassification,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    LlamaModelForwards.llama_for_sequence_classification_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=LlamaForSequenceClassification,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        layers = super().get_held_layers()
        layers.append(self.model.model.rotary_emb)

        return layers
