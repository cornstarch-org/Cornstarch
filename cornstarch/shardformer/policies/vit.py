from __future__ import annotations

import functools
import itertools
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
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal
from transformers.models.vit.modeling_vit import ViTConfig, ViTModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.vit import ViTAttentionForwards, ViTModelForwards
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class ViTModelPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(config, ViTConfig), "config must be an instance of ViTConfig"
        config: ViTConfig = cast(ViTConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend([f"encoder.layer.{i}" for i in range(config.num_hidden_layers)])
        modules.extend(["layernorm", "pooler"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.vit.modeling_vit" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "ViTModel" else "vit."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embeddings" not in modules_in_template[0]:
            raise ValueError("embeddings must be in the first stage.")

        if not all(
            module in template.modules_per_stage[-1]
            for module in [f"{prefix}layernorm", f"{prefix}pooler"]
        ):
            raise ValueError("layernorm and pooler must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.vit.modeling_vit import (
            ViTLayer,
            ViTSelfAttention,
        )

        config: ViTConfig = self.model.config
        policy = {}

        tp_size = self.shard_config.tensor_parallel_size
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        if self.shard_config.enable_tensor_parallelism:
            hidden_size //= tp_size

            assert (
                num_heads % tp_size == 0
            ), "The number of attention heads must be divisible by the tensor parallel size."
            num_heads //= tp_size

        attention_attribute_replacement = {
            "num_attention_heads": num_heads,
            "all_head_size": hidden_size,
            "attention_probs_dropout_prob": config.attention_probs_dropout_prob,
        }

        policy[ViTSelfAttention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": functools.partial(
                    ViTAttentionForwards.forward,
                    shard_config=self.shard_config,
                )
            },
        )

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

            policy[ViTModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                },
            )

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        if self.shard_config.enable_tensor_parallelism:
            policy[ViTLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate.dense",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                ]
            )

        if self.shard_config.enable_fused_normalization:
            self.append_or_create_method_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layernorm_before",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layernorm_after",
                        target_module=FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=ViTLayer,
            )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="layernorm",
                    )
                ],
                policy=policy,
                target_key=ViTModel,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    ViTModelForwards.vit_model_forward, shard_config=self.shard_config
                )
            },
            policy=policy,
            target_key=ViTModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: ViTModel
        if self.model.__class__.__name__ == "ViTModel":
            module = self.model
        else:
            module = self.model.vit
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.extend([module.layernorm, module.pooler])

        return held_layers
