import functools
import itertools
import warnings
from typing import Callable, Dict, List, cast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Row,
    PaddingEmbedding,
    PaddingLMHead,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from torch import Tensor, nn
from transformers import PretrainedConfig
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import Phi3Model

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.phi3 import Phi3Forwards, Phi3PipelineForwards
from cornstarch.shardformer.modeling.qkv_fused_linear import FusedLinear1D_Col
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Phi3Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, Phi3Config
        ), "config must be an instance of Phi3Config"
        config: Phi3Config = cast(Phi3Config, config)

        modules = []
        modules.extend(["embed_tokens", "embed_dropout"])
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.phi3.modeling_phi3" in template.model_name
        ), "The pipeline template is not for Phi3 model."

        prefix = "" if self.model.__class__.__name__ == "Phi3Model" else "model."
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(
            module in modules_in_template[0]
            for module in [f"{prefix}embed_tokens", f"{prefix}embed_dropout"]
        ):
            raise ValueError("The embedding layers must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("The norm layer must be in the last stage.")

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: dict
    ):
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "Phi3Model":
            module = self.model
        else:
            module = self.model.model

        layers_per_stage = stage_manager.distribute_layers(
            len(module.layers), stage_manager.num_stages
        )
        stage_index = stage_manager.get_stage_index(
            layers_per_stage, stage_manager.stage
        )
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

        module: Phi3Model
        if self.model.__class__.__name__ == "Phi3Model":
            module = self.model
        else:
            module = self.model.model
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = stage_manager.distribute_layers(len(module.layers))
        if stage_manager.is_first_stage():
            held_layers.extend([module.embed_tokens, module.embed_dropout])
        start_idx, end_idx = stage_manager.get_stage_index(layers_per_stage)
        held_layers.extend(module.layers[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.norm)
        return held_layers

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.phi3.modeling_phi3 import (
            Phi3Attention,
            Phi3DecoderLayer,
            Phi3FlashAttention2,
            Phi3SdpaAttention,
        )

        policy = {}

        if self.shard_config.enable_sequence_parallelism:
            self.shard_config.enable_sequence_parallelism = False
            warnings.warn(
                "Phi3 doesn't support sequence parallelism now, will ignore the sequence parallelism flag."
            )

        config: Phi3Config = self.model.config
        if self.shard_config.enable_tensor_parallelism:
            assert (
                config.num_attention_heads % self.shard_config.tensor_parallel_size == 0
            ), "The number of attention heads must be divisible by tensor parallel size."
            assert (
                config.num_key_value_heads % self.shard_config.tensor_parallel_size == 0
            ), "The number of key_value heads must be divisible by tensor parallel size."

            decoder_attribute_replacement = {
                "self_attn.hidden_size": config.hidden_size
                // self.shard_config.tensor_parallel_size,
                "self_attn.num_heads": config.num_attention_heads
                // self.shard_config.tensor_parallel_size,
            }
            if getattr(config, "num_key_value_heads", False):
                decoder_attribute_replacement["self_attn.num_key_value_heads"] = (
                    config.num_key_value_heads // self.shard_config.tensor_parallel_size
                )

            policy[Phi3DecoderLayer] = ModulePolicyDescription(
                attribute_replacement=decoder_attribute_replacement,
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.qkv_proj",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(n_fused=3),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_up_proj",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(n_fused=2),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                    ),
                ],
            )

        embedding_cls = None
        if self.shard_config.enable_tensor_parallelism:
            embedding_cls = VocabParallelEmbedding1D
        elif self.tie_weight:
            embedding_cls = PaddingEmbedding

        if embedding_cls is not None:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="embed_tokens",
                    target_module=embedding_cls,
                    kwargs={
                        "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
                    },
                ),
                policy=policy,
                target_key=Phi3Model,
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
                target_key=Phi3DecoderLayer,
            )

            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="norm",
                    target_module=FusedRMSNorm,
                ),
                policy=policy,
                target_key=Phi3Model,
            )

        if self.shard_config.enable_flash_attention:
            ATTN_IMPLEMENTATION = {
                "eager": Phi3Attention,
                "sdpa": Phi3SdpaAttention,
                "flash_attention_2": Phi3FlashAttention2,
            }
            attn_cls = ATTN_IMPLEMENTATION[config._attn_implementation]
            self.append_or_create_method_replacement(
                description={
                    "forward": Phi3Forwards.phi3_flash_attention_forward,
                },
                policy=policy,
                target_key=attn_cls,
            )
            if self.pipeline_stage_manager is None:
                # replace mistral model forward method
                self.append_or_create_method_replacement(
                    description={
                        "forward": functools.partial(
                            Phi3Forwards.phi3_model_forward_for_flash_attention,
                            shard_config=self.shard_config,
                        )
                    },
                    policy=policy,
                    target_key=Phi3Model,
                )

        return policy


class Phi3ModelPolicy(Phi3Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return Phi3Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=Phi3Model,
                new_forward=Phi3PipelineForwards.phi3_model_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        return []


class Phi3ForCausalLMPolicy(Phi3Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"model.{module}" for module in Phi3Policy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("The lm_head layer must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            target_module = VocabParallelLMHead1D
            kwargs = {
                "gather_output": not self.shard_config.parallel_output,
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
            }
            methods_replacement = {
                "forward": functools.partial(
                    Phi3Forwards.phi3_for_causal_lm_forward_with_dist_cross_entropy,
                    shard_config=self.shard_config,
                )
            }
        else:
            target_module = PaddingLMHead
            kwargs = {
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
            }
            methods_replacement = None

        policy.update(
            {
                Phi3ForCausalLM: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=target_module,
                            kwargs=kwargs,
                        )
                    ],
                    method_replacement=methods_replacement,
                )
            }
        )

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=Phi3ForCausalLM,
                new_forward=Phi3PipelineForwards.phi3_for_causal_lm_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        phi3_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if (
                id(phi3_model.embed_tokens.weight) == id(self.model.lm_head.weight)
                and self.pipeline_stage_manager.num_stages > 1
            ):
                # tie weights
                return [
                    {
                        0: phi3_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.lm_head.weight,
                    }
                ]
        return []
