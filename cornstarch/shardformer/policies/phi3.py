import functools
import itertools
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Row,
    PaddingEmbedding,
    PaddingLMHead,
    VocabParallelEmbedding1D,
    VocabParallelLMHead1D,
)
from colossalai.shardformer.layer.qkv_fused_linear import FusedLinear1D_Col
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
from cornstarch.shardformer.modeling.phi3 import (
    Phi3AttentionForwards,
    Phi3ModelForwards,
)
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
        modules.extend(["embed_tokens", "rotary_emb"])
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
            for module in [f"{prefix}embed_tokens", f"{prefix}rotary_emb"]
        ):
            raise ValueError("Teh embedding layers must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("The norm layer must be in the last stage.")

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
            held_layers.extend([module.embed_tokens, module.rotary_emb])
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
            Phi3RMSNorm,
        )

        config: Phi3Config = self.model.config
        policy = {}

        # This is to avoid refererence to its weight which has been replaced by a placeholder
        policy[Phi3RMSNorm] = ModulePolicyDescription(
            method_replacement={
                "extra_repr": lambda self: f"eps={self.variance_epsilon}"
            }
        )

        sp_mode = self.shard_config.sequence_parallelism_mode or None
        sp_size = self.shard_config.sequence_parallel_size or None
        if sp_mode == "ring_attn" and not self.is_causal:
            raise ValueError(
                "Ring attention is only meant for causal language modeling."
            )

        tp_size = self.shard_config.tensor_parallel_size
        num_q_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        hidden_size = config.hidden_size

        if sp_mode == "all_to_all":
            # Ulysses all-to-all context parallelism needs to partition number of heads
            hidden_size //= sp_size

            assert (
                num_q_heads % sp_size == 0
            ), "The number of attention heads must be divisible by the sequence parallel size."
            num_q_heads //= sp_size

            assert (
                num_kv_heads % sp_size == 0
            ), "The number of key_value heads must be divisible by the sequence parallel size."
            num_kv_heads //= sp_size

        if self.shard_config.enable_tensor_parallelism:
            hidden_size //= tp_size

            assert (
                num_q_heads % tp_size == 0
            ), "The number of attention heads must be divisible by the tensor parallel size."
            num_q_heads //= tp_size

            assert (
                num_kv_heads % tp_size == 0
            ), "The number of key_value heads must be divisible by the tensor parallel size."
            num_kv_heads //= tp_size

        attention_attribute_replacement = {}
        attention_attribute_replacement["hidden_size"] = hidden_size
        attention_attribute_replacement["num_heads"] = num_q_heads
        attention_attribute_replacement["num_key_value_heads"] = num_kv_heads

        policy[Phi3Attention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": functools.partial(
                    Phi3AttentionForwards.forward,
                    shard_config=self.shard_config,
                )
            },
        )

        if self.shard_config.enable_flash_attention:
            policy[Phi3Model] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        if self.shard_config.enable_tensor_parallelism:
            policy[Phi3DecoderLayer] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="self_attn.qkv_proj",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(
                            split_sizes=[config.hidden_size] * 3,
                            seq_parallel_mode=sp_mode,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="self_attn.o_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.gate_up_proj",
                        target_module=FusedLinear1D_Col,
                        kwargs=dict(
                            split_sizes=[config.intermediate_size] * 2,
                            seq_parallel_mode=sp_mode,
                        ),
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.down_proj",
                        target_module=Linear1D_Row,
                        kwargs=dict(seq_parallel_mode=sp_mode),
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

        return policy


class Phi3ModelPolicy(Phi3Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return Phi3Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Phi3ModelForwards.phi3_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Phi3Model,
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

        self.is_causal = True
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            # add a new item for causal lm
            target_module = VocabParallelLMHead1D
            kwargs = {
                "gather_output": not self.shard_config.parallel_output,
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by,
            }
        else:
            target_module = PaddingLMHead
            kwargs = {
                "make_vocab_size_divisible_by": self.shard_config.make_vocab_size_divisible_by
            }

        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="lm_head",
                target_module=target_module,
                kwargs=kwargs,
            ),
            policy=policy,
            target_key=Phi3ForCausalLM,
        )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Phi3ModelForwards.phi3_for_causal_lm_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Phi3ForCausalLM,
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
