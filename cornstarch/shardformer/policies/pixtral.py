import functools
import itertools
import warnings
from typing import cast

import torch.nn as nn
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
from transformers import PretrainedConfig
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.pixtral import PixtralVisionModelForwards
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class PixtralVisionModelPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, PixtralVisionConfig
        ), f"config must be PixtralVisionConfig, got {type(config)}"
        config: PixtralVisionConfig = cast(PixtralVisionConfig, config)

        modules = []
        modules.extend(["patch_conv", "ln_pre", "patch_positional_embedding"])
        modules.extend(
            [f"transformer.layers.{i}" for i in range(config.num_hidden_layers)]
        )

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.pixtral.modeling_pixtral" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        assert hasattr(self.model, "config"), "Model must have a config attribute."
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(
            module in modules_in_template[0]
            for module in ["patch_conv", "ln_pre", "patch_positional_embedding"]
        ):
            raise ValueError(
                "The conv, layernorm, and positional embedding must be in the first stage."
            )

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.pixtral.modeling_pixtral import (
            PixtralAttention,
            PixtralAttentionLayer,
        )

        config: PixtralVisionConfig = self.model.config
        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "PixtralVisionModel doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_flash_attention:
            self.shard_config.enable_flash_attention = False
            warnings.warn(
                "PixtralVisionModel doesn't support flash attention now, will ignore the flash attention flag."
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

        policy[PixtralAttention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement
        )

        if self.shard_config.enable_tensor_parallelism:
            policy[PixtralAttentionLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.q_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.k_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.v_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.gate_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward.down_proj",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

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
                target_key=PixtralAttentionLayer,
            )

            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="ln_pre",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=PixtralVisionModel,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    PixtralVisionModelForwards.pixtral_vision_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=PixtralVisionModel,
        )

        return policy

    def get_held_layers(self) -> list[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: PixtralVisionModel = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(
            len(module.transformer.layers)
        )
        if stage_manager.is_first_stage():
            held_layers.append(module.patch_conv)
            held_layers.append(module.ln_pre)
            held_layers.append(module.patch_positional_embedding)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.transformer.layers[start_idx:end_idx])

        return held_layers
