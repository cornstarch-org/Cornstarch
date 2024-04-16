import itertools
import warnings
from typing import Callable, Dict, List, cast

from colossalai.shardformer.layer import (
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
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.clip import (
    get_clip_flash_attention_forward,
    get_clip_naive_attention_forward,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class CLIPVisionTransformerPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, CLIPVisionConfig
        ), f"config must be CLIPVisionConfig, got {type(config)}"
        config: CLIPVisionConfig = cast(CLIPVisionConfig, config)

        modules = []
        modules.extend(["embeddings", "pre_layrnorm"])
        modules.extend([f"encoder.layer.{i}" for i in range(config.num_hidden_layers)])
        modules.append("post_layernorm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.clip.modeling_clip" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = (
            ""
            if self.model.__class__.__name__ == "CLIPVisionTransformer"
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
        if f"{prefix}pre_layrnorm" not in modules_in_template[0]:
            raise ValueError("pre layernorm must be in the first stage.")

        if f"{prefix}post_layernorm" not in modules_in_template[-1]:
            raise ValueError("post layernorm must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.clip.modeling_clip import (
            CLIPAttention,
            CLIPEncoderLayer,
            CLIPVisionTransformer,
        )

        policy: dict[str | nn.Module, ModulePolicyDescription] = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = FusedLayerNorm
        else:
            norm_cls = LayerNorm

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "CLIP doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[CLIPEncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.num_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.embed_dim": self.model.config.hidden_size
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

        # handle CLIPEncoderLayer layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="layer_norm1",
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="layer_norm2",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=CLIPEncoderLayer,
        )

        # handle CLIPVisionTransformer layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="pre_layrnorm",  # typo in HF impl
                    target_module=norm_cls,
                ),
                SubModuleReplacementDescription(
                    suffix="post_layernorm",
                    target_module=norm_cls,
                ),
            ],
            policy=policy,
            target_key=CLIPVisionTransformer,
        )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_clip_flash_attention_forward(),
                },
                policy=policy,
                target_key=CLIPAttention,
            )
        else:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_clip_naive_attention_forward(),
                },
                policy=policy,
                target_key=CLIPAttention,
            )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            pass

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        raise NotImplementedError

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        raise NotImplementedError

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ):
        raise NotImplementedError


class CLIPVisionModelPolicy(CLIPVisionTransformerPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"vision_model.{module}"
            for module in CLIPVisionTransformerPolicy.get_all_modules(config)
        ]
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.clip.modeling_clip import CLIPVisionTransformer

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=CLIPVisionTransformer,
                new_forward=None,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in clip model"""
        return []
