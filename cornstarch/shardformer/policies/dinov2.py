import itertools
from functools import partial
from typing import Callable, cast

from colossalai.shardformer.layer import (
    DropoutForParallelInput,
    DropoutForReplicatedInput,
    Linear1D_Col,
    Linear1D_Row,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from torch import nn
from torch.nn.modules import Module
from transformers import PretrainedConfig
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.dinov2 import Dinov2PipelineForwards
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
        )

        policy = {}

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
                attribute_replacement={
                    "attention.attention.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "attention.attention.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
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
                        suffix="mlp.weights_in"
                        if self.model.config.use_swiglu_ffn
                        else "mlp.fc1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.weights_out"
                        if self.model.config.use_swiglu_ffn
                        else "mlp.fc2",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        # use fused operator
        if self.shard_config.enable_fused_normalization:
            raise NotImplementedError

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
        layers_per_stage = self.distribute_layers(
            len(module.encoder.layer), stage_manager.num_stages
        )
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.encoder.layer[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.layernorm)

        return held_layers

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: dict
    ):
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ in ["Dinov2Model", "Dinov2Backbone"]:
            module = self.model
        else:
            module = self.model.dinov2

        layers_per_stage = self.distribute_layers(
            len(module.encoder.layer), stage_manager.num_stages
        )
        stage_index = self.get_stage_index(layers_per_stage, stage_manager.stage)
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


class Dinov2ModelPolicy(Dinov2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return Dinov2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(self, template)

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=Dinov2Model,
                new_forward=Dinov2PipelineForwards.dinov2_model_forward,
                policy=policy,
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            raise NotImplementedError

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            raise NotImplementedError

        return policy


class Dinov2ForImageClassificationPolicy(Dinov2ModelPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"dinov2.{module}" for module in Dinov2Policy.get_all_modules(config)
        ]
        modules.append("classifier")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "classifier" not in template.modules_per_stage[-1]:
            raise ValueError("classifier must be in the last stage.")

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        from transformers.models.dinov2.modeling_dinov2 import (
            Dinov2ForImageClassification,
            Dinov2Model,
        )

        policy = super().module_policy()
        if self.shard_config.enable_tensor_parallelism:
            new_item = {
                Dinov2ForImageClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="classifier",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True),
                        )
                    ]
                )
            }
            policy.update(new_item)

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=Dinov2Model,
                new_forward=Dinov2PipelineForwards.dinov2_model_forward,
                policy=policy,
            )
            self.set_pipeline_forward(
                model_cls=Dinov2ForImageClassification,
                new_forward=Dinov2PipelineForwards.dinov2_for_image_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> list[Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.classifier)
        return held_layers


class Dinov2BackbonePolicy(Dinov2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return Dinov2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Backbone

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=Dinov2Backbone,
                new_forward=Dinov2PipelineForwards.dinov2_backbone_forward,
                policy=policy,
            )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            raise NotImplementedError

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            raise NotImplementedError

        return policy
