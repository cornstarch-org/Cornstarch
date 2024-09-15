import warnings
from typing import Dict

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
from transformers.models.qwen2_audio.configuration_qwen2_audio import (
    Qwen2AudioEncoderConfig,
)
from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

from cornstarch.shardformer.modeling.qwen2_audio import Qwen2AudioForwards


class Qwen2AudioEncoderPolicy(Policy):
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

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Qwen2Audio doesn't support sequence parallelism, will ignore the sequence parallelism flag."
            )

        config: Qwen2AudioEncoderConfig = self.model.config
        if self.shard_config.enable_tensor_parallelism:
            assert (
                config.encoder_attention_heads % self.shard_config.tensor_parallel_size
                == 0
            ), "The number of attention heads must be divisible by tensor parallel size."

            policy[Qwen2AudioEncoderLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "self_attn.embed_dim": config.d_model
                    // self.shard_config.tensor_parallel_size,
                    "self_attn.num_heads": config.encoder_attention_heads
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

        if self.shard_config.enable_flash_attention:
            ATTN_IMPLEMENTATION = {
                "eager": Qwen2AudioAttention,
                "sdpa": Qwen2AudioSdpaAttention,
                "flash_attention_2": Qwen2AudioFlashAttention2,
            }
            attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]
            self.append_or_create_method_replacement(
                description={
                    "forward": Qwen2AudioForwards.qwen2_audio_flash_attention_forward
                },
                policy=policy,
                target_key=attn_cls,
            )

        if self.shard_config.enable_jit_fused:
            warnings.warn(
                "Qwen2Audio doesn't support JIT fusion, will ignore the flag."
            )
            self.shard_config.enable_jit_fused = False

        return policy
