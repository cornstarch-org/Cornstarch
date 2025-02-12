import itertools
import warnings
from functools import partial
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedLayerNorm,
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
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.siglip import SiglipVisionModelForwards
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class SiglipVisionTransformerPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, SiglipVisionConfig
        ), f"config must be an instance of SiglipVisionConfig, got {type(config)}"
        config: SiglipVisionConfig = cast(SiglipVisionConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend(f"encoder.layers.{i}" for i in range(config.num_hidden_layers))
        modules.append("post_layernorm")
        modules.append("head")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.siglip.modeling_siglip" in template.model_name
        ), "The pipeline template is not for Siglip model."

        prefix = (
            ""
            if self.model.__class__.__name__ == "SiglipVisionTransformer"
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
            raise ValueError("post_layernorm must be in the last stage.")

        if f"{prefix}head" not in modules_in_template[-1]:
            raise ValueError("head must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.siglip.modeling_siglip import (
            SiglipAttention,
            SiglipEncoderLayer,
            SiglipFlashAttention2,
            SiglipMultiheadAttentionPoolingHead,
            SiglipSdpaAttention,
        )

        config: SiglipVisionConfig = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": SiglipAttention,
            "sdpa": SiglipSdpaAttention,
            "flash_attention_2": SiglipFlashAttention2,
        }
        attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "CLIP doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
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

        policy[attn_cls] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
        )

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )
            attention_attribute_replacement["is_causal"] = False

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=attention_attribute_replacement,
                method_replacement={"forward": (SiglipFlashAttention2.forward)},
            )

            policy[SiglipVisionTransformer] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[SiglipEncoderLayer] = ModulePolicyDescription(
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

            policy[SiglipMultiheadAttentionPoolingHead] = ModulePolicyDescription(
                sub_module_replacement=[
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
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm2",
                        target_module=FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=SiglipEncoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="post_layernorm",
                    target_module=FusedLayerNorm,
                ),
                policy=policy,
                target_key=SiglipVisionTransformer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layernorm",
                    target_module=FusedLayerNorm,
                ),
                policy=policy,
                target_key=SiglipMultiheadAttentionPoolingHead,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": SiglipVisionModelForwards.siglip_vision_transformer_forward,
            },
            policy=policy,
            target_key=SiglipVisionTransformer,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: SiglipVisionTransformer
        if self.model.__class__.__name__ == "SiglipVisionTransformer":
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
            held_layers.append(module.head)

        return held_layers


class SiglipVisionModelPolicy(SiglipVisionTransformerPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"vision_model.{module}"
            for module in SiglipVisionTransformerPolicy.get_all_modules(config)
        ]

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        return super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.siglip.modeling_siglip import (
            SiglipVisionModel,
        )

        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": partial(
                    SiglipVisionModelForwards.siglip_vision_model_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=SiglipVisionModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()
