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
from transformers import PretrainedConfig
from transformers.models.phi4_multimodal.configuration_phi4_multimodal import (
    Phi4MultimodalAudioConfig,
)
from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
    Phi4MultimodalAudioModel,
)

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.phi4_multimodal import (
    Phi4MultimodalAudioAttentionForwards,
    Phi4MultimodalForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Phi4MultimodalAudioModelPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, Phi4MultimodalAudioConfig
        ), f"config must be Phi4MultimodalAudioConfig, got {type(config)}"
        config: Phi4MultimodalAudioConfig = cast(Phi4MultimodalAudioConfig, config)

        modules = []
        modules.extend(["encoder_embedding", "embed", "relative_attention_bias_layer"])
        modules.extend([f"encoders.{i}" for i in range(config.num_blocks)])

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.phi4_multimodal.modeling_phi4_multimodal"
            in template.model_name
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
            for module in [
                "encoder_embedding",
                "embed",
                "relative_attention_bias_layer",
            ]
        ):
            raise ValueError(
                "The first stage must contain encoder_embedding, embed, and relative_attention_bias_layer."
            )

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import (
            Phi4MultimodalAudioAttention,
            Phi4MultimodalAudioConformerEncoderLayer,
            Phi4MultimodalAudioModel,
        )

        config: Phi4MultimodalAudioConfig = self.model.config
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

        attention_attribute_replacement = {}
        attention_attribute_replacement["head_dim"] = hidden_size // num_heads

        policy[Phi4MultimodalAudioAttention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": functools.partial(
                    Phi4MultimodalAudioAttentionForwards.forward,
                    shard_config=self.shard_config,
                )
            },
        )

        if self.shard_config.enable_flash_attention:
            policy[Phi4MultimodalAudioModel] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2",
                }
            )

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        if self.shard_config.enable_tensor_parallelism:
            policy[Phi4MultimodalAudioConformerEncoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.q_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.k_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.v_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward_in.gate_up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward_in.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward_out.gate_up_proj",
                        target_module=Linear1D_Col,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward_out.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                ],
            )

        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="feed_forward_in.layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="conv.layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="feed_forward_out.layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm_att",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=Phi4MultimodalAudioConformerEncoderLayer,
            )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Phi4MultimodalForwards.phi4_multimodal_audio_model_forward,
                    shard_config=self.shard_config,
                ),
            },
            policy=policy,
            target_key=Phi4MultimodalAudioModel,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: Phi4MultimodalAudioModel
        if self.model.__class__.__name__ == "Phi4MultimodalAudioEmbedding":
            module = self.model.encoder
        else:
            module = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.encoders))
        if stage_manager.is_first_stage():
            held_layers.extend(
                [
                    module.encoder_embedding,
                    module.embed,
                    module.relative_attention_bias_layer,
                ]
            )
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.encoders[start_idx:end_idx])

        return held_layers
