import warnings

from colossalai.shardformer.layer import (
    DropoutForParallelInput,
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
from torch import nn

from cornstarch.shardformer.modeling.clip import (
    get_clip_flash_attention_forward,
    get_clip_naive_attention_forward,
)


class CLIPVisionPolicy(Policy):
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

        # # add missing dropout in attention
        # self.append_or_create_submodule_replacement(
        #     description=[
        #         SubModuleReplacementDescription(
        #             suffix="dropout",
        #             target_module=DropoutForParallelInput,
        #             kwargs=dict(p=self.model.config.attention_dropout),
        #         ),
        #     ],
        #     policy=policy,
        #     target_key=CLIPAttention,
        # )

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
