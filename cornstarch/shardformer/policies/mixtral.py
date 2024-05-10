from __future__ import annotations

import itertools
import warnings
from functools import partial
from typing import Callable, Dict, List, cast

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
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.models.mixtral import MixtralConfig, MixtralModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.mixtral import (
    MixtralForwards,
    get_lm_forward_with_dist_cross_entropy,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

__all__ = ["MixtralPolicy", "MixtralForCausalLMPolicy"]


class MixtralPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config,
            MixtralConfig,
        ), "config must be an instance of MixtralConfig"
        config: MixtralConfig = cast(MixtralConfig, config)

        modules = []
        modules.append("embed_tokens")
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.mixtral.modeling_mixtral" in template.model_name
        ), "The pipeline template is not for the model that the polict is designed for."

        prefix = "" if self.model.__class__.__name__ == "MixtralModel" else "model."

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

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: dict
    ) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager is None:
            return

        module: MixtralModel
        if self.model.__class__.__name__ == "MixtralModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        layers_per_stage = stage_manager.distribute_layers(
            len(module.layers), stage_manager.num_stages
        )
        stage_index = stage_manager.get_stage_index(
            layers_per_stage, stage_manager.stage
        )
        method_replacement = {
            "forward": partial(
                new_forward,
                stage_manager=stage_manager,
                stage_index=stage_index,
                shard_config=self.shard_config,
            )
        }
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=model_cls
        )

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: MixtralModel
        if self.model.__class__.__name__ == "MixtralModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(
            len(module.layers), stage_manager.num_stages
        )

        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)

        return held_layers

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralBlockSparseTop2MLP,
            MixtralDecoderLayer,
            MixtralModel,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Mistral doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="embed_tokens",
                target_module=VocabParallelEmbedding1D
                if self.shard_config.enable_tensor_parallelism
                else PaddingEmbedding,
                kwargs={
                    "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
                },
            ),
            policy=policy,
            target_key=MixtralModel,
        )

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_key_value_heads": self.model.config.num_key_value_heads
                // self.shard_config.tensor_parallel_size,
            }

            policy[MixtralDecoderLayer] = ModulePolicyDescription(
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
                ],
            )

            policy[MixtralBlockSparseTop2MLP] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="w1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="w3",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="w2",
                        target_module=Linear1D_Row,
                    ),
                ]
            )

        # optimize configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_method_replacement(
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
                target_key=MixtralDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=MixtralModel,
            )

        if self.shard_config.enable_flash_attention:
            pass

        return policy


class MixtralModelPolicy(MixtralPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return MixtralPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=MixtralModel,
                new_forward=MixtralForwards.mixtral_model_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class MixtralForCausalLMPolicy(MixtralPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"model.{module}" for module in MixtralPolicy.get_all_modules(config)
        ]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            new_item = {
                MixtralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=VocabParallelLMHead1D,
                            kwargs=dict(
                                gather_output=not self.shard_config.parallel_output,
                                make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by,
                            ),
                        ),
                    ]
                )
            }
            if self.shard_config.parallel_output:
                new_item[MixtralForCausalLM].method_replacement = {
                    "forward": get_lm_forward_with_dist_cross_entropy(self.shard_config)
                }
        else:
            new_item = {
                MixtralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=PaddingLMHead,
                            kwargs=dict(
                                make_vocab_size_divisible_by=self.shard_config.make_vocab_size_divisible_by
                            ),
                        )
                    ]
                )
            }
        policy.update(new_item)

        if self.pipeline_stage_manager:
            # set None as default
            self.set_pipeline_forward(
                model_cls=MixtralForCausalLM,
                new_forward=MixtralForwards.mixtral_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        mixtral_model: MixtralModel = self.model.model
        if (
            self.pipeline_stage_manager
            and self.pipeline_stage_manager.num_stages > 1
            and id(mixtral_model.embed_tokens.weight) == id(self.model.lm_head.weight)
        ):
            # tei weights
            return [
                {
                    0: mixtral_model.embed_tokens.weight,
                    self.pipeline_stage_manager.num_stages
                    - 1: self.model.lm_head.weight,
                }
            ]
        return []
