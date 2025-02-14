import functools
import itertools
import warnings
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedLayerNorm,
    FusedLinear1D_Col,
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
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.qwen2_vision import (
    Qwen2VisionModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Qwen2VisionTransformerPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, Qwen2VLVisionConfig
        ), f"config must be Qwen2VLVisionConfig, got {type(config)}"
        config: Qwen2VLVisionConfig = cast(Qwen2VLVisionConfig, config)

        modules = []
        modules.extend(["patch_embed", "rotary_pos_emb"])
        modules.extend([f"blocks.{i}" for i in range(config.depth)])
        modules.append("merger")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.qwen2_vl.modeling_qwen2_vl" in template.model_name
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
            for module in ["patch_embed", "rotary_pos_emb"]
        ):
            raise ValueError(
                "The first stage of the pipeline template must contain 'patch_embed' and 'rotary_pos_emb'."
            )

        if "merge" not in modules_in_template[-1]:
            raise ValueError(
                "The last stage of the pipeline template must contain 'merge'."
            )

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            PatchMerger,
            Qwen2VisionTransformerPretrainedModel,
            Qwen2VLVisionBlock,
            VisionAttention,
            VisionFlashAttention2,
            VisionSdpaAttention,
        )

        config: Qwen2VLVisionConfig = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": VisionAttention,
            "sdpa": VisionSdpaAttention,
            "flash_attention_2": VisionFlashAttention2,
        }
        attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Qwen2Vision doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_heads = config.num_heads
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
            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=attention_attribute_replacement,
                method_replacement={"forward": VisionFlashAttention2.forward},
            )

            policy[Qwen2VisionTransformerPretrainedModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[Qwen2VLVisionBlock] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.qkv",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(split_sizes=[config.embed_dim] * 3),
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

            policy[PatchMerger] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="mlp.0",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.2",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="norm1",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="norm2",
                        target_module=FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=Qwen2VLVisionBlock,
            )

            self.append_or_create_method_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="ln_q",
                        target_module=FusedLayerNorm,
                    )
                ],
                policy=policy,
                target_key=PatchMerger,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Qwen2VisionModelForwards.qwen2_vision_transformer_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Qwen2VisionTransformerPretrainedModel,
        )

        return policy

    def get_held_layers(self) -> list[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: Qwen2VisionTransformerPretrainedModel = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.blocks))
        if stage_manager.is_first_stage():
            held_layers.extend([module.patch_embed, module.rotary_pos_emb])
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.blocks[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.merger)

        return held_layers
