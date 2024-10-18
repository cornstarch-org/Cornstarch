import functools
import itertools
from typing import Dict, List, cast

import torch.nn as nn
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
    Policy,
    SubModuleReplacementDescription,
)
from torch import Tensor
from transformers import PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal

from cornstarch.models.internlm2 import InternLM2Config, InternLM2Model
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.internlm2 import (
    InternLM2AttentionForwards,
    InternLM2ModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class InternLM2Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, InternLM2Config
        ), "config must be an instance of InternLM2Config"
        config: InternLM2Config = cast(InternLM2Config, config)

        modules = []
        modules.append("tok_embeddings")
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "cornstarch.models.internlm.modeling_internlm2" in template.model_name
        ), "The pipeline template is not for InternLM2 model."

        prefix = "" if self.model.__class__.__name__ == "InternLM2Model" else "model."
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}tok_embeddings" not in modules_in_template[0]:
            raise ValueError("tok_embeddings must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: InternLM2Model
        if self.model.__class__.__name__ == "InternLM2Model":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.tok_embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)
        return held_layers

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.internlm2.modeling_internlm2 import (
            InternLM2Attention,
            InternLM2DecoderLayer,
            InternLM2FlashAttention2,
            InternLM2SdpaAttention,
        )

        config: InternLM2Config = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": InternLM2Attention,
            "sdpa": InternLM2SdpaAttention,
            "flash_attention_2": InternLM2FlashAttention2,
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
                    InternLM2AttentionForwards.forward,
                    shard_config=self.shard_config,
                ),
            },
        )

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

            policy[InternLM2Model] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                },
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[InternLM2DecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.wqkv",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.wo",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.w1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.w3",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.w2",
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
                    suffix="tok_embeddings",
                    target_module=embedding_cls,
                    kwargs={
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
                    },
                ),
                policy=policy,
                target_key=InternLM2Model,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="attention_norm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="ffn_norm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=InternLM2DecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="norm",
                        target_module=FusedRMSNorm,
                    )
                ],
                policy=policy,
                target_key=InternLM2Model,
            )

        return policy


class InternLM2ModelPolicy(InternLM2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return InternLM2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    InternLM2ModelForwards.internlm2_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=InternLM2Model,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class InternLM2ForCausalLMPolicy(InternLM2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"model.{module}" for module in InternLM2Policy.get_all_modules(config)
        ]
        modules.append("output")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("The lm_head layer must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.internlm2.modeling_internlm2 import InternLM2ForCausalLM

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
                suffix="output",
                target_module=target_module,
                kwargs=kwargs,
            ),
            policy=policy,
            target_key=InternLM2ForCausalLM,
        )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    InternLM2ModelForwards.internlm2_for_causal_lm_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=InternLM2ForCausalLM,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.output)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        internlm2_model = self.model.model
        if (
            self.pipeline_stage_manager
            and self.pipeline_stage_manager.num_stages > 1
            and id(internlm2_model.tok_embeddings.weight)
            == id(self.model.output.weight)
        ):
            # tie weights
            return [
                {
                    0: internlm2_model.tok_embeddings.weight,
                    self.pipeline_stage_manager.num_stages
                    - 1: self.model.output.weight,
                }
            ]
        return []
