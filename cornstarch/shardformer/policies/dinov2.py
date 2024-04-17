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
from torch.nn.modules import Module


class Dinov2Policy(Policy):
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


class Dinov2ModelPolicy(Dinov2Policy):
    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()

        # use flash attention
        if self.shard_config.enable_flash_attention:
            raise NotImplementedError

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            raise NotImplementedError

        return policy


class Dinov2ForImageClassificationPolicy(Dinov2ModelPolicy):
    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        from transformers.models.dinov2.modeling_dinov2 import (
            Dinov2ForImageClassification,
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

        return policy


class Dinov2BackbonePolicy(Dinov2Policy):
    def module_policy(self) -> dict[str | Module, ModulePolicyDescription]:
        policy = super().module_policy()

        # use flash attention
        if self.shard_config.enable_flash_attention:
            raise NotImplementedError

        # use jit fused operator
        if self.shard_config.enable_jit_fused:
            raise NotImplementedError

        return policy
