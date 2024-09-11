import functools
import warnings
from typing import Callable, Dict, List

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

from cornstarch.models.internlm import InternLM2Config
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.internlm2 import InternLM2Forwards, InternLM2Model
from cornstarch.shardformer.modeling.qkv_fused_linear import FusedLinear1D_Col
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class InternLM2Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        raise NotImplementedError

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        raise NotImplementedError

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: dict
    ):
        raise NotImplementedError

    def get_held_layers(self) -> List[nn.Module]:
        raise NotImplementedError

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.internlm.modeling_internlm2 import (
            InternLM2Attention,
            InternLM2DecoderLayer,
            InternLM2FlashAttention2,
            InternLM2SdpaAttention,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "InternLM2 doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        config: InternLM2Config = self.model.config
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
                "attention.hidden_size": config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "attention.num_heads": config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
            }
            if getattr(config, "num_key_value_heads", False):
                decoder_attribute_replacement["attention.num_key_value_heads"] = (
                    config.num_key_value_heads // self.shard_config.tensor_parallel_size
                )

            policy[InternLM2DecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.wqkv",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(n_fused=3),
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
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=InternLM2Model,
            )

        if self.shard_config.enable_flash_attention:
            ATTN_IMPLEMENTATION = {
                "eager": InternLM2Attention,
                "sdpa": InternLM2SdpaAttention,
                "flash": InternLM2FlashAttention2,
            }
            attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]
            self.append_or_create_method_replacement(
                description={
                    "forward": InternLM2Forwards.internlm2_flash_attention_forward,
                },
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                # replace internlm2 model forward method
                self.append_or_create_method_replacement(
                    description={
                        "forward": functools.partial(
                            InternLM2Forwards.internlm2_model_forward_for_flash_attention,
                            shard_config=self.shard_config,
                        )
                    },
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

        if self.pipeline_stage_manager:
            raise NotImplementedError

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class InternLM2ForCausalPolicy(InternLM2Policy):
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
        from cornstarch.models.internlm.modeling_internlm2 import InternLM2ForCausalLM

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
                    InternLM2Forwards.internlm2_for_causal_lm_forward_with_dist_cross_entropy,
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
                InternLM2ForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=SubModuleReplacementDescription(
                        suffix="lm_head",
                        target_module=target_module,
                        kwargs=kwargs,
                    ),
                    method_replacement=methods_replacement,
                )
            }
        )

        if self.pipeline_stage_manager:
            raise NotImplementedError

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
            == id(internlm2_model.output.weight)
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
