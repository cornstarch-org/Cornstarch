import functools
import itertools
import warnings
from typing import cast

from colossalai.shardformer.layer import (
    DropoutForParallelInput,
    DropoutForReplicatedInput,
    FusedLayerNorm,
    LayerNorm,
    Linear1D_Col,
    Linear1D_Row,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from torch.nn.modules import Module
from transformers import PretrainedConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.dinov2 import (
    Dinov2ModelForwards,
    Dinov2SelfAttentionForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Dinov2Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, Dinov2Config
        ), "config must be an instance of Dinov2Config"
        config: Dinov2Config = cast(Dinov2Config, config)

        modules = []
        modules.append("embeddings")
        modules.extend([f"encoder.layer.{i}" for i in range(config.num_hidden_layers)])
        modules.append("layernorm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.dinov2.modeling_dinov2" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = (
            ""
            if self.model.__class__.__name__ in ["Dinov2Model", "Dinov2Backbone"]
            else "dinov2."
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

        if f"{prefix}layernorm" not in modules_in_template[-1]:
            raise ValueError("layernorm must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self):
        return self.model

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        from transformers.models.dinov2.modeling_dinov2 import (
            Dinov2Embeddings,
            Dinov2Layer,
            Dinov2Model,
            Dinov2SdpaSelfAttention,
            Dinov2SelfAttention,
        )

        config: Dinov2Config = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": Dinov2SelfAttention,
            "sdpa": Dinov2SdpaSelfAttention,
        }
        attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Dinov2 doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
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
        attention_attribute_replacement["num_attention_heads"] = num_heads
        attention_attribute_replacement["all_head_size"] = hidden_size
        attention_attribute_replacement["attention_probs_dropout_prob"] = (
            config.attention_probs_dropout_prob
        )

        policy[attn_cls] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": (
                    Dinov2SelfAttentionForwards.flash_attention_forward
                    if self.shard_config.enable_flash_attention
                    else Dinov2SelfAttentionForwards.sdpa_forward
                ),
            },
        )

        if self.shard_config.enable_tensor_parallelism:
            policy[Dinov2Embeddings] = ModulePolicyDescription(
                attribute_replacement={},
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=DropoutForReplicatedInput,
                    )
                ],
            )

            policy[Dinov2Layer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.attention.query",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.key",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.value",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.attention.dropout",
                        target_module=DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=DropoutForReplicatedInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix=(
                            "mlp.weights_in"
                            if self.model.config.use_swiglu_ffn
                            else "mlp.fc1"
                        ),
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix=(
                            "mlp.weights_out"
                            if self.model.config.use_swiglu_ffn
                            else "mlp.fc2"
                        ),
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        if self.shard_config.enable_fused_normalization:
            norm_cls = FusedLayerNorm
        else:
            norm_cls = LayerNorm

        # use fused operator
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="norm1",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="norm2",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=Dinov2Layer,
        )
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="layernorm",
                target_module=norm_cls,
            ),
            policy=policy,
            target_key=Dinov2Model,
        )

        return policy

    def postprocess(self) -> Module:
        return self.model

    def get_held_layers(self) -> list[Module]:
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ in ["Dinov2Model", "Dinov2Backbone"]:
            module = self.model
        else:
            module = self.model.dinov2
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layer))
        if stage_manager.is_first_stage(ignore_chunk=True):
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(module.layernorm)

        return held_layers


class Dinov2ModelPolicy(Dinov2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return Dinov2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(self, template)

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Dinov2ModelForwards.dinov2_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Dinov2Model,
        )

        return policy


class Dinov2BackbonePolicy(Dinov2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = Dinov2Policy.get_all_modules(config)
        modules.append("layernorm")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="layernorm",
                target_module=(
                    FusedLayerNorm
                    if self.shard_config.enable_fused_normalization
                    else LayerNorm
                ),
            ),
            policy=policy,
            target_key=Dinov2Backbone,
        )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Dinov2ModelForwards.dinov2_backbone_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Dinov2Backbone,
        )

        return policy
