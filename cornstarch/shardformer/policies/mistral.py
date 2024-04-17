"""
Code modified after copied from:
https://github.com/hpcaitech/ColossalAI/blob/v0.3.6/colossalai/shardformer/policies/mistral.py
"""

import warnings

import torch.nn as nn
from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
    Linear1D_Row,
    VocabParallelEmbedding1D,
)
from colossalai.shardformer.modeling.mistral import get_mistral_flash_attention_forward
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)

from cornstarch.shardformer.policies.utils import (
    resize_embeddings,
    resize_lm_head,
)


class MistralPolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self):
        # Resize embedding
        vocab_size = self.model.config.vocab_size
        world_size = self.shard_config.tensor_parallel_size

        if self.shard_config.enable_tensor_parallelism and vocab_size % world_size != 0:
            new_vocab_size = vocab_size + world_size - vocab_size % world_size

            embeddings: nn.Embedding = self.model.get_input_embeddings()
            if embeddings.num_embeddings * world_size == new_vocab_size:
                # Skip if the embedding layer has already been resized
                return self.model

            resize_embeddings(new_vocab_size, embeddings)

            lm_head: nn.Embedding | nn.Linear = self.model.get_output_embeddings()
            if isinstance(lm_head, nn.Embedding):
                resize_embeddings(new_vocab_size, lm_head)
            elif isinstance(lm_head, nn.Linear):
                resize_lm_head(new_vocab_size, lm_head)

            self.model.vocab_size = new_vocab_size

        return self.model

    def module_policy(self) -> dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention,
            MistralDecoderLayer,
            MistralModel,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Mistral doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        if self.shard_config.enable_tensor_parallelism:
            decoder_attribute_replacement = {
                "self_attn.hidden_size": self.model.config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": self.model.config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_key_value_heads": self.model.config.num_key_value_heads
                // self.shard_config.tensor_parallel_size,
            }

            policy[MistralDecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
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
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.up_proj",
                        target_module=Linear1D_Col,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=VocabParallelEmbedding1D,
                ),
                policy=policy,
                target_key=MistralModel,
            )

        # optimization configuration
        if self.shard_config.enable_fused_normalization:
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="input_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_attention_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=MistralDecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=MistralModel,
            )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_mistral_flash_attention_forward(),
                },
                policy=policy,
                target_key=MistralAttention,
            )

        return policy

    def postprocess(self):
        return self.model


class MistralModelPolicy(MistralPolicy):
    def __init__(self) -> None:
        super().__init__()

    def module_policy(self):
        if self.pipeline_stage_manager:
            warnings.warn("Mistral doesn't support pipeline parallelism now.")

        return super().module_policy()


class MistralForCausalLMPolicy(MistralPolicy):
    def module_policy(self):
        from transformers import MistralForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for casual lm
            new_item = {
                MistralForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True),
                        )
                    ]
                )
            }

            if self.pipeline_stage_manager:
                warnings.warn("Mistral doesn't support pipeline parallelism now.")

            policy.update(new_item)

        return policy


class MistralForSequenceClassificationPolicy(MistralPolicy):
    def module_policy(self):
        from transformers import MistralForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for sequence classification
            new_item = {
                MistralForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="score",
                            target_module=Linear1D_Col,
                            kwargs=dict(gather_output=True),
                        )
                    ]
                )
            }

            if self.pipeline_stage_manager:
                warnings.warn("Mistral doesn't support pipeline parallelism now.")

            policy.update(new_item)
        return policy
