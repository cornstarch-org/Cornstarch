from __future__ import annotations

import functools
import itertools
import warnings
from typing import Dict, cast

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
from torch import nn
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.llama import LlamaForwards, LlamaPipelineForwards
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

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn("Cornstarch llama doesn't support sequence parallelism.")

        if self.shard_config.enable_tensor_parallelism:
            assert (
                self.model.config.num_attention_heads
                % self.shard_config.tensor_parallel_size
                == 0
            ), "The number of attention heads must be divisible by tensor parallel size."
            if hasattr(self.model.config, "num_key_value_heads"):
                assert (
                    self.model.config.num_key_value_heads
                    >= self.shard_config.tensor_parallel_size
                    and self.model.config.num_key_value_heads
                    % self.shard_config.tensor_parallel_size
                    == 0
                ), "The number of key_value heads must be divisible by, and must not be less than tensor parallel size."
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
            }
            if getattr(self.model.config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    self.model.config.num_key_value_heads
                    // self.shard_config.tensor_parallel_size
                )

            policy[LlamaDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
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

        if self.shard_config.enable_flash_attention:
            ATTN_IMPLEMENTATION = {
                "eager": LlamaAttention,
                "flash_attention_2": LlamaFlashAttention2,
                "sdpa": LlamaSdpaAttention,
            }
            attn_cls = ATTN_IMPLEMENTATION[self.model.config._attn_implementation]
            self.append_or_create_method_replacement(
                description={"forward": LlamaForwards.llama_flash_attention_forward},
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                # replace llama model forward method
                self.append_or_create_method_replacement(
                    description={
                        "forward": functools.partial(
                            LlamaForwards.llama_model_forward_for_flash_attention,
                            shard_config=self.shard_config,
                        ),
                    },
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

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=LlamaModel,
                new_forward=LlamaPipelineForwards.llama_model_forward,
                policy=policy,
            )

        return policy


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

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            target_module = VocabParallelLMHead1D
            kwargs = {
                "gather_output": not self.shard_config.parallel_output,
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
            }
            methods_replacement = {
                "forward": functools.partial(
                    LlamaForwards.llama_for_causal_lm_forward_with_dist_cross_entropy,
                    shard_config=self.shard_config,
                )
            }
        else:
            target_module = PaddingLMHead
            kwargs = {
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
            }
            methods_replacement = None

        policy.update(
            {
                LlamaForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=target_module,
                            kwargs=kwargs,
                        )
                    ],
                    method_replacement=methods_replacement,
                )
            }
        )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=LlamaForCausalLM,
                new_forward=LlamaPipelineForwards.llama_for_causal_lm_forward,
                policy=policy,
            )

        return policy


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
            policy.update(
                {
                    LlamaForSequenceClassification: ModulePolicyDescription(
                        sub_module_replacement=[
                            SubModuleReplacementDescription(
                                suffix="score",
                                target_module=Linear1D_Col,
                                kwargs=dict(gather_output=True),
                            )
                        ],
                    )
                }
            )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=LlamaForSequenceClassification,
                new_forward=LlamaPipelineForwards.llama_for_sequence_classification_forward,
                policy=policy,
            )

        return policy
