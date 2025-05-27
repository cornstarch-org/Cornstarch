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
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextModel,
    Gemma3ForCausalLM,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.gemma3 import (
    # Gemma3AttentionForwards,
    Gemma3ModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Gemma3Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, Gemma3TextConfig
        ), f"config must be of type Gemma3TextConfig, but got {type(config)}"
        config: Gemma3TextConfig = cast(Gemma3TextConfig, config)

        modules = []
        modules.extend(["embed_tokens", "rotary_emb", "rotary_emb_local"])
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "trasnformers.models.gemma3.modeling_gemma3" in template.model_name
        ), "The pipeline template is not for Gemma3 model"

        prefix = "" if self.model.__class__.__name__ == "Gemma3Model" else "model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                f"modules in the template do not match the modules in the model. "
                f"Expected: {modules}, Got: {modules_in_template}"
            )

        if not all(module in modules_in_template[0] for module in modules):
            raise ValueError("The pipeline template is not for Gemma3 model.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None
        module: Gemma3TextModel
        if self.model.__class__.__name__ == "Gemma3TextModel":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.extend(
                [module.embed_tokens, module.rotary_emb, module.rotary_emb_local]
            )
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)
        return held_layers

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3Attention,
            Gemma3RMSNorm,
            Gemma3DecoderLayer,
        )

        config: Gemma3TextConfig = self.model.config
        policy = {}

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        if sp_mode is not None:
            raise NotImplementedError(
                "Sequence parallelism is not supported for Gemma3 model."
            )

        # This is to avoid refererence to its weight which has been replaced by a placeholder
        policy[Gemma3RMSNorm] = ModulePolicyDescription(
            method_replacement={
                "extra_repr": lambda self: f"eps={self.variance_epsilon}"
            }
        )

        if self.shard_config.enable_flash_attention:
            policy[Gemma3TextModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[Gemma3DecoderLayer] = ModulePolicyDescription(
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
                    SubModuleReplacementDescription(
                        suffix="pre_feedforward_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_feedforward_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_norm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_norm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=Gemma3DecoderLayer,
            )

        return policy


class Gemma3TextModelPolicy(Gemma3Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return Gemma3Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Gemma3ModelForwards.gemma3_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Gemma3TextModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()


class Gemma3ForCausalLMPolicy(Gemma3Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"model.{module}" for module in Gemma3Policy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Gemma3ModelForwards.gemma3_for_causal_lm_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Gemma3ForCausalLM,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        gemma_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(gemma_model.embed_tokens.weight) == id(self.model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: gemma_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.lm_head.weight,
                    }
                ]

        return []
