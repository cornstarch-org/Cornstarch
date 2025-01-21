import functools
import itertools
import warnings
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedLinear1D_Col,
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

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig
from cornstarch.models.intern_vit.modeling_intern_vit import InternVisionModel
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.intern_vit import (
    InternVisionAttentionForwards,
    InternVisionModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class InternVisionModelPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, InternVisionConfig
        ), "config must be an instance of InternVisionConfig"
        config: InternVisionConfig = cast(InternVisionConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend(f"encoder.layers.{i}" for i in range(config.num_hidden_layers))

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "cornstarch.models.intern_vit.modeling_intern_vit" in template.model_name
        ), "The pipeline template is not for Cornstarch intern vit model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if "embeddings" not in modules_in_template[0]:
            raise ValueError("embeddings must be in the first stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.models.intern_vit.modeling_intern_vit import (
            InternAttention,
            InternVisionEncoderLayer,
        )

        config: InternVisionConfig = self.model.config

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "InternVision doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size

        if self.shard_config.enable_tensor_parallelism:
            hidden_size = hidden_size // tp_size

            assert (
                num_heads % tp_size == 0
            ), "The number of attention heads must be divisible by the tensor parallel size."
            num_heads //= tp_size

            assert (
                not config.qk_normalization
            ), "QK normalization is not supported with tensor parallelism."

        attention_attribute_replacement = {}
        attention_attribute_replacement["embed_dim"] = hidden_size
        attention_attribute_replacement["num_heads"] = num_heads

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

        policy[InternAttention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": (
                    InternVisionAttentionForwards.flash_attention_forward
                    if self.shard_config.enable_flash_attention
                    else InternVisionAttentionForwards.eager_attention_forward
                ),
            },
        )

        if self.shard_config.enable_tensor_parallelism:
            policy[InternVisionEncoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.qkv",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(split_sizes=[config.hidden_size] * 3),
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.proj",
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
                        suffix="norm1",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="norm2",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=InternVisionEncoderLayer,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    InternVisionModelForwards.intern_vit_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=InternVisionModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: InternVisionModel = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layers[start_idx:end_idx])

        return held_layers
