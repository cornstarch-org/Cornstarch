import functools
import itertools
import warnings
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from cornstarch.models.evaclip.modeling_evaclip import (
    EvaCLIPVisionModel,
    EvaCLIPVisionTransformer,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.evaclip import (
    EvaCLIPAttentionForwards,
    EvaCLIPModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class EvaCLIPVisionPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, CLIPVisionConfig
        ), f"config must be CLIPVisionConfig, got {type(config)}"
        config: CLIPVisionConfig = cast(CLIPVisionConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend([f"encoder.layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("post_layernorm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "cornstarch.models.evaclip.modeling_evaclip" in template.model_name
        ), "The pipeline template is not for cornstarch evaclip model."

        prefix = (
            ""
            if self.model.__class__.__name__ == "EvaCLIPVisionTransformer"
            else "vision_model."
        )

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embeddings" not in modules_in_template[0]:
            raise ValueError("embeddings must be in the first stage.")

        if f"{prefix}post_layernorm" not in modules_in_template[-1]:
            raise ValueError("post layernorm must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: EvaCLIPVisionTransformer
        if self.model.__class__.__name__ == "EvaCLIPVisionTransformer":
            module = self.model
        else:
            module = self.model.vision_model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.post_layernorm)
        return held_layers

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.evaclip.modeling_evaclip import (
            EvaCLIPAttention,
            EvaCLIPEncoderLayer,
            EvaCLIPVisionTransformer,
        )

        config: CLIPVisionConfig = self.model.config
        policy: dict[str | nn.Module, ModulePolicyDescription] = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "EvaCLIP doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        if self.shard_config.enable_tensor_parallelism:
            hidden_size //= tp_size

            assert (
                num_heads % tp_size == 0
            ), "The number of attention heads must be divisible by the tensor parallel size."
            num_heads //= tp_size

        attention_attribute_replacement = {}
        attention_attribute_replacement["embed_dim"] = hidden_size
        attention_attribute_replacement["num_heads"] = num_heads

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

        policy[EvaCLIPAttention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": (
                    EvaCLIPAttentionForwards.flash_attention_forward
                    if self.shard_config.enable_flash_attention
                    else EvaCLIPAttentionForwards.sdpa_forward
                )
            },
        )

        if self.shard_config.enable_tensor_parallelism:
            policy[EvaCLIPEncoderLayer] = ModulePolicyDescription(
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
                        suffix="self_attn.out_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.fc2",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layer_norm1",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm2",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=EvaCLIPEncoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="post_layernorm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=EvaCLIPVisionTransformer,
            )

        return policy


class EvaCLIPVisionTransformerPolicy(EvaCLIPVisionPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return EvaCLIPVisionPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    EvaCLIPModelForwards.eva_clip_vision_transformer_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=EvaCLIPVisionTransformer,
        )

        return policy


class EvaCLIPVisionModelPolicy(EvaCLIPVisionPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"vision_model.{module}"
            for module in EvaCLIPVisionPolicy.get_all_modules(config)
        ]

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    EvaCLIPModelForwards.evaclip_vision_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=EvaCLIPVisionModel,
        )

        return policy
