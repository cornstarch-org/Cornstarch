import functools
import itertools
import warnings
from typing import Dict, List, cast

import torch.nn as nn
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
from transformers import PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal
from transformers.models.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.qwen2_audio import Qwen2AudioModelForwards
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Qwen2AudioEncoderPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, Qwen2AudioEncoderConfig
        ), f"config must be Qwen2AudioEncoderConfig, got {type(config)}"
        config: Qwen2AudioEncoderConfig = cast(Qwen2AudioEncoderConfig, config)

        modules = []
        modules.extend(["conv1", "conv2", "embed_positions"])
        modules.extend([f"layers.{i}" for i in range(config.encoder_layers)])
        modules.extend(["layer_norm", "avg_pooler"])

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.qwen2_audio.modeling_qwen2_audio"
            in template.model_name
        ), "The pipeline template is not for Qwen2Audio."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(
            module in modules_in_template[0]
            for module in ["conv1", "conv2", "embed_positions"]
        ):
            raise ValueError("convolutions and embeddings must be in the first stage.")

        if not all(
            module in modules_in_template[-1] for module in ["layer_norm", "avg_pooler"]
        ):
            raise ValueError("layer norm and avg pooler must be in the last stage.")

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: Qwen2AudioEncoder = self.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.extend([module.conv1, module.conv2, module.embed_positions])
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.extend([module.layer_norm, module.avg_pooler])

        return held_layers

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.qwen2_audio.modeling_qwen2_audio import (
            Qwen2AudioAttention,
            Qwen2AudioEncoderLayer,
            Qwen2AudioFlashAttention2,
            Qwen2AudioSdpaAttention,
        )

        config: Qwen2AudioEncoderConfig = self.model.config
        ATTN_IMPLEMENTATION = {
            "eager": Qwen2AudioAttention,
            "sdpa": Qwen2AudioSdpaAttention,
            "flash_attention_2": Qwen2AudioFlashAttention2,
        }
        attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Qwen2Audio doesn't support sequence parallelism, will ignore the sequence parallelism flag."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_heads = config.encoder_attention_heads
        hidden_size = config.d_model

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
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

            policy[attn_cls] = ModulePolicyDescription(
                attribute_replacement=attention_attribute_replacement,
                method_replacement={"forward": Qwen2AudioFlashAttention2.forward},
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[Qwen2AudioEncoderLayer] = ModulePolicyDescription(
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
                        suffix="fc1",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="fc2",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="self_attn_layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="final_layer_norm",
                        target_module=FusedLayerNorm,
                    ),
                ],
                policy=policy,
                target_key=Qwen2AudioEncoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="layer_norm",
                    target_module=FusedLayerNorm,
                ),
                policy=policy,
                target_key=Qwen2AudioEncoder,
            )

        if self.shard_config.enable_jit_fused:
            warnings.warn(
                "Qwen2Audio doesn't support JIT fusion, will ignore the flag."
            )
            self.shard_config.enable_jit_fused = False

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Qwen2AudioModelForwards.qwen2_audio_encoder_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Qwen2AudioEncoder,
        )

        return policy
