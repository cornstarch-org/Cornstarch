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
    Policy,
    SubModuleReplacementDescription,
)
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.qwen3_moe import (
    Qwen3MoeModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Qwen3MoePolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, Qwen3MoeConfig
        ), "config must be an instance of Qwen3MoeConfig"
        config: Qwen3MoeConfig = cast(Qwen3MoeConfig, config)

        modules = []
        modules.extend(["embed_tokens", "rotary_emb"])
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.qwen3_moe.modeling_qwen3_moe" in template.model_name
        ), "The pipeline template is not for Qwen3Moe model."

        prefix = "" if self.model.__class__.__name__ == "Qwen3MoeModel" else "model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(module in modules_in_template[0] for module in modules):
            raise ValueError("The pipeline template is not for Qwen3Moe model.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: Qwen3MoeModel
        if self.model.__class__.__name__ == "Qwen3MoeModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embed_tokens)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)
        held_layers.append(module.rotary_emb)
        return held_layers

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeMLP,
            Qwen3MoeRMSNorm,
            Qwen3MoeDecoderLayer,
        )

        config: Qwen3MoeConfig = self.model.config
        policy = {}

        # This is to avoid refererence to its weight which has been replaced by a placeholder
        policy[Qwen3MoeRMSNorm] = ModulePolicyDescription(
            method_replacement={
                "extra_repr": lambda self: f"eps={self.variance_epsilon}"
            }
        )

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        if sp_mode is not None:
            raise NotImplementedError(
                "Sequence parallelism is not supported for Qwen3Moe model."
            )

        if self.shard_config.enable_flash_attention:
            policy[Qwen3MoeModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[Qwen3MoeDecoderLayer] = ModulePolicyDescription(
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
                ]
            )

            policy[Qwen3MoeMLP] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="gate_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                ]
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="norm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=Qwen3MoeDecoderLayer,
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
                target_key=Qwen3MoeDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=Qwen3MoeModel,
            )

        return policy

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def config_sanity_check(self):
        pass


class Qwen3MoeModelPolicy(Qwen3MoePolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return Qwen3MoePolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(self, template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Qwen3MoeModelForwards.qwen3_moe_model_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=Qwen3MoeModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()


class Qwen3MoeForCausalLMPolicy(Qwen3MoePolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = Qwen3MoePolicy.get_all_modules(config)
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        self.is_causal = True
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Qwen3MoeModelForwards.qwen3_moe_for_causal_lm_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=Qwen3MoeForCausalLM,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers
