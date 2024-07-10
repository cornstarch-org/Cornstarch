import functools
import itertools
import warnings
from typing import Callable, Dict, List, cast

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

from cornstarch.models.intern_vit.configuration_intern_vit import InternVisionConfig
from cornstarch.models.intern_vit.modeling_intern_vit import InternVisionModel
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.intern_vit import (
    InternVisionForwards,
    InternVisionPipelineForwards,
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

        policy: dict[str | nn.Module, ModulePolicyDescription] = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "InternVision doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        config: InternVisionConfig = self.model.config
        if self.shard_config.enable_tensor_parallelism:
            assert (
                config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), (
                f"The number of attention heads {config.num_attention_heads} must be divisible "
                f"by tensor parallel size {self.shard_config.tensor_parallel_size}."
            )
            assert (
                not config.qk_normalization
            ), "QK normalization is not supported with tensor parallelism."

            attribute_replacement = {
                "attn.num_heads": config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
                "attn.embed_dim": config.hidden_size
                // self.shard_config.tensor_parallel_size,
            }

            policy[InternVisionEncoderLayer] = ModulePolicyDescription(
                attribute_replacement=attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.qkv",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(n_fused=3),
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

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": InternVisionForwards.intern_flash_attention_forward
                },
                policy=policy,
                target_key=InternAttention,
            )
        else:
            self.append_or_create_method_replacement(
                description={
                    "forward": InternVisionForwards.intern_eager_attention_forward,
                },
                policy=policy,
                target_key=InternAttention,
            )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=InternVisionModel,
                new_forward=InternVisionPipelineForwards.intern_vit_model_forward,
                policy=policy,
            )

        return policy

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ):
        assert self.pipeline_stage_manager is not None

        module: InternVisionModel = self.model
        stage_manager = self.pipeline_stage_manager
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layers))
        stage_index = stage_manager.get_stage_index(layers_per_stage)

        method_replacement = {
            "forward": functools.partial(
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

        module: InternVisionModel = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoder.layers))
        if stage_manager.is_first_stage():
            held_layers.append(module.embeddings)
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoder.layers[start_idx:end_idx])

        return held_layers
