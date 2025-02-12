import functools
import itertools
from typing import Dict, List, cast

from colossalai.shardformer.layer import (
    FusedRMSNorm,
    Linear1D_Col,
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
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM, Gemma2Model

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.modeling.gemma2 import (
    Gemma2AttentionForwards,
    Gemma2ModelForwards,
)
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class Gemma2Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, Gemma2Config
        ), f"config must be of type Gemma2Config, but got {type(config)}"
        config: Gemma2Config = cast(Gemma2Config, config)

        modules = []
        modules.extend(["embed_tokens", "rotary_emb"])
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "trasnformers.models.gemma2.modeling_gemma2" in template.model_name
        ), "The pipeline template is not for Gemma2 model"

        prefix = "" if self.model.__class__.__name__ == "Gemma2Model" else "model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                f"modules in the template do not match the modules in the model. "
                f"Expected: {modules}, Got: {modules_in_template}"
            )

        if f"{prefix}embed_tokens" not in modules_in_template[0]:
            raise ValueError("embed_tokens must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        self.tie_weight = self.tie_weight_check()
        return self.model

    def postprocess(self) -> nn.Module:
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        assert self.pipeline_stage_manager is not None

        module: Gemma2Model
        if self.model.__class__.__name__ == "Gemma2Model":
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

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from transformers.models.gemma2.modeling_gemma2 import (
            Gemma2Attention,
            Gemma2DecoderLayer,
            Gemma2RMSNorm,
        )

        config: Gemma2Config = self.model.config
        policy = {}

        # This is to avoid refererence to its weight which has been replaced by a placeholder
        policy[Gemma2RMSNorm] = ModulePolicyDescription(
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
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        hidden_size = config.hidden_size

        if sp_mode == "all_to_all":
            # Ulysses all-to-all context parallelism needs to partition number of heads
            hidden_size //= sp_size

            assert (
                num_q_heads % sp_size == 0
            ), "The number of attention heads must be divisible by the sequence parallel size."
            num_q_heads //= sp_size

            if num_kv_heads:
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

            if num_kv_heads:
                assert (
                    num_kv_heads % tp_size == 0
                ), "The number of key_value heads must be divisible by the tensor parallel size."
                num_kv_heads //= tp_size

        attention_attribute_replacement = {}
        attention_attribute_replacement["hidden_size"] = hidden_size
        attention_attribute_replacement["num_heads"] = num_q_heads
        if num_kv_heads:
            attention_attribute_replacement["num_key_value_heads"] = num_kv_heads

        if self.shard_config.enable_flash_attention:
            attention_attribute_replacement["_flash_attn_uses_top_left_mask"] = (
                not is_flash_attn_greater_or_equal("2.1.0")
            )

            policy[Gemma2Model] = ModulePolicyDescription(
                attribute_replacement={
                    "config._attn_implementation": "flash_attention_2"
                }
            )

        policy[Gemma2Attention] = ModulePolicyDescription(
            attribute_replacement=attention_attribute_replacement,
            method_replacement={
                "forward": functools.partial(
                    Gemma2AttentionForwards.forward,
                    shard_config=self.shard_config,
                )
            },
        )

        if self.shard_config.enable_tensor_parallelism:
            policy[Gemma2DecoderLayer] = ModulePolicyDescription(
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
                target_key=Gemma2Model,
            )

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
                    SubModuleReplacementDescription(
                        suffix="pre_feedforward_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                    SubModuleReplacementDescription(
                        suffix="post_feedforward_layernorm",
                        target_module=FusedRMSNorm,
                    ),
                ],
                policy=policy,
                target_key=Gemma2DecoderLayer,
            )
            self.append_or_create_submodule_replacement(
                description=[
                    SubModuleReplacementDescription(
                        suffix="norm",
                        target_module=FusedRMSNorm,
                    )
                ],
                policy=policy,
                target_key=Gemma2Model,
            )

        return policy


class Gemma2ModelPolicy(Gemma2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return Gemma2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        policy = super().module_policy()

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Gemma2ModelForwards.gemma2_model_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Gemma2Model,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()


class Gemma2ForCausalLMPolicy(Gemma2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"model.{module}" for module in Gemma2Policy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        self.is_causal = True
        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
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
            target_key=Gemma2ForCausalLM,
        )

        self.append_or_create_method_replacement(
            description={
                "forward": functools.partial(
                    Gemma2ModelForwards.gemma2_for_causal_lm_forward,
                    shard_config=self.shard_config,
                )
            },
            policy=policy,
            target_key=Gemma2ForCausalLM,
        )

        return policy

    def get_held_layers(self) -> List[nn.Module]:
        stage_manager = self.pipeline_stage_manager
        held_layers = super().get_held_layers()
        if stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        gemma_model = self.model.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(gemma_model.embed_tokens.weight) == id(self.model.lm_head.weight):
                # tie weights
                return [
                    {
                        0: gemma_model.embed_tokens.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.lm_head.weight,
                    }
                ]

        return []
