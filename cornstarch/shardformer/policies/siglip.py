import itertools
import warnings
from functools import partial
from typing import Callable, Dict, List, cast

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
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.siglip import (
    SiglipVisionForwards,
    SiglipVisionPipelineForwards,
)
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
            SiglipMultiheadAttentionPoolingHead,
        )

        policy: dict[str | nn.Module, ModulePolicyDescription] = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "CLIP doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        config: SiglipVisionConfig = cast(SiglipVisionConfig, self.model.config)
        if self.shard_config.enable_tensor_parallelism:
            assert (
                config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), (
                f"The number of attention heads {config.num_attention_heads} must be divisible "
                f"by tensor parallel size {self.shard_config.tensor_parallel_size}."
            )

            policy[SiglipEncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.num_heads": config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.embed_dim": config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                },
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

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": SiglipVisionForwards.clip_flash_attention_forward,
                },
                policy=policy,
                target_key=SiglipAttention,
            )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=SiglipVisionTransformer,
                new_forward=SiglipVisionPipelineForwards.siglip_vision_transformer_forward,
                policy=policy,
            )

        return policy

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ):
        if self.pipeline_stage_manager is None:
            return

        module: SiglipVisionTransformer
        if self.model.__class__.__name__ == "SiglipVisionTransformer":
            module = self.model
        else:
            module = self.model.vision_model

        stage_manager = self.pipeline_stage_manager
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layers))
        stage_index = stage_manager.get_stage_index(layers_per_stage)

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

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=SiglipVisionModel,
                new_forward=SiglipVisionPipelineForwards.siglip_vision_model_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()
