# Copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/gpt2.py

from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, Dict, List, cast

import colossalai.shardformer.layer as col_nn
from colossalai.shardformer.modeling.gpt2 import (
    GPT2PipelineForwards,
    get_gpt2_flash_attention_forward,
    gpt2_sequence_parallel_forward_fn,
)
from colossalai.shardformer.policies.base_policy import (
    ModulePolicyDescription,
    Policy,
    SubModuleReplacementDescription,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)
from cornstarch.shardformer.policies.utils import (
    resize_embeddings,
    resize_lm_head,
)
from torch import Tensor, nn
from transformers import GPT2Config, PretrainedConfig

__all__ = [
    "GPT2Policy",
    "GPT2ModelPolicy",
    "GPT2LMHeadModelPolicy",
    "GPT2DoubleHeadsModelPolicy",
    "GPT2ForTokenClassificationPolicy",
    "GPT2ForSequenceClassificationPolicy",
]


class GPT2Policy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        assert isinstance(
            config, GPT2Config
        ), "config must be an instance of GPT2Config"
        config: GPT2Config = cast(GPT2Config, config)

        modules = []
        modules.extend(["wte", "wpe", "drop"])
        modules.extend([f"h.{i}" for i in range(config.n_layer)])
        modules.append("ln_f")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.gpt2.modeling_gpt2" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "GPT2Model" else "transformer."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(
            itertools.chain.from_iterable(template.modules_per_stage)
        )
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(
            module in template.modules_per_stage[0]
            for module in [f"{prefix}wte", f"{prefix}wpe", f"{prefix}drop"]
        ):
            raise ValueError("wte, wpe, and drop must be in the first stage.")

        if f"{prefix}ln_f" not in template.modules_per_stage[-1]:
            raise ValueError("ln_f must be in the last stage.")

    def config_sanity_check(self):
        pass

    def preprocess(self):
        r"""
        Reshape the Embedding layer to make the embedding dimension divisible by world_size
        """
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

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import (
            GPT2Attention,
            GPT2Block,
            GPT2Model,
        )

        policy = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm
        use_sequence_parallel = self.shard_config.enable_sequence_parallelism
        overlap = self.shard_config.enable_sequence_overlap
        if self.shard_config.enable_tensor_parallelism:
            policy[GPT2Model] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="wte",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="drop",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ]
            )

            policy[GPT2Block] = ModulePolicyDescription(
                attribute_replacement={
                    "attn.embed_dim": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "attn.split_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "attn.num_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attn.c_attn",
                        target_module=col_nn.GPT2FusedLinearConv1D_Col,
                        kwargs={
                            "n_fused": 3,
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.c_proj",
                        target_module=col_nn.GPT2FusedLinearConv1D_Row,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.c_fc",
                        target_module=col_nn.GPT2FusedLinearConv1D_Col,
                        kwargs={
                            "n_fused": 1,
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.c_proj",
                        target_module=col_nn.GPT2FusedLinearConv1D_Row,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.attn_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attn.resid_dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="mlp.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

        # optimization configuration
        self.append_or_create_submodule_replacement(
            description=SubModuleReplacementDescription(
                suffix="ln_f",
                target_module=norm_cls,
            ),
            policy=policy,
            target_key=GPT2Model,
        )

        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="ln_1",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
                SubModuleReplacementDescription(
                    suffix="ln_2",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
                SubModuleReplacementDescription(
                    suffix="ln_cross_attn",
                    target_module=norm_cls,
                    ignore_if_not_exist=True,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
            ],
            policy=policy,
            target_key=GPT2Block,
        )

        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_gpt2_flash_attention_forward(),
                },
                policy=policy,
                target_key=GPT2Attention,
            )

        if self.shard_config.enable_sequence_parallelism:
            policy[GPT2Model].method_replacement = {
                "forward": gpt2_sequence_parallel_forward_fn(self.shard_config)
            }

        return policy

    def postprocess(self):
        return self.model

    def get_held_layers(self) -> List[nn.Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        layers_per_stage = self.distribute_layers(
            len(module.h), stage_manager.num_stages
        )
        if stage_manager.is_first_stage():
            held_layers.append(module.wte)
            held_layers.append(module.wpe)
            held_layers.append(module.drop)
        start_idx, end_idx = self.get_stage_index(layers_per_stage, stage_manager.stage)
        held_layers.extend(module.h[start_idx:end_idx])
        if stage_manager.is_last_stage():
            held_layers.append(module.ln_f)
        return held_layers

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ) -> None:
        """If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy."""
        if not self.pipeline_stage_manager:
            raise ValueError(
                "set_pipeline_forward method can only be called when pipeline parallel is enabled."
            )
        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "GPT2Model":
            module = self.model
        else:
            module = self.model.transformer

        layers_per_stage = self.distribute_layers(
            len(module.h), stage_manager.num_stages
        )
        stage_index = self.get_stage_index(layers_per_stage, stage_manager.stage)
        method_replacement = {
            "forward": partial(
                new_forward,
                stage_manager=stage_manager,
                stage_index=stage_index,
                shard_config=self.shard_config,
            )
        }
        self.append_or_create_method_replacement(
            description=method_replacement, policy=policy, target_key=model_cls
        )


# GPT2Model
class GPT2ModelPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return GPT2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model

        policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2Model,
                new_forward=GPT2PipelineForwards.gpt2_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[nn.Module]:
        return super().get_held_layers()

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2Model."""
        return []


# GPT2LMHeadModel
class GPT2LMHeadModelPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2LMHeadModel: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=col_nn.Linear1D_Col,
                            kwargs={"gather_output": True},
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2LMHeadModel,
                new_forward=GPT2PipelineForwards.gpt2_lmhead_model_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.lm_head)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """The weights of wte and lm_head are shared."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None:
            if stage_manager.num_stages > 1 and id(module.transformer.wte.weight) == id(
                module.lm_head.weight
            ):
                first_stage, last_stage = 0, stage_manager.num_stages - 1
                return [
                    {
                        first_stage: module.transformer.wte.weight,
                        last_stage: module.lm_head.weight,
                    }
                ]
        return []


# GPT2DoubleHeadsModel
class GPT2DoubleHeadsModelPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.extend(["lm_head", "multiple_choice_head"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["lm_head", "multiple_choice_head"]
        ):
            raise ValueError(
                "lm_head and multiple_choice_head must be in the last stage."
            )

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModel

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2DoubleHeadsModel: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="lm_head",
                            target_module=col_nn.Linear1D_Col,
                            kwargs={"gather_output": True},
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2DoubleHeadsModel,
                new_forward=GPT2PipelineForwards.gpt2_double_heads_model_forward,
                policy=module_policy,
            )

        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            multiple_choice_head = self.model.multiple_choice_head
            held_layers.append(self.model.lm_head)
            held_layers.append(multiple_choice_head.summary)
            held_layers.append(multiple_choice_head.activation)
            held_layers.append(multiple_choice_head.first_dropout)
            held_layers.append(multiple_choice_head.last_dropout)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """The weights of wte and lm_head are shared."""
        module = self.model
        stage_manager = self.pipeline_stage_manager
        if stage_manager is not None:
            if stage_manager.num_stages > 1 and id(module.transformer.wte.weight) == id(
                module.lm_head.weight
            ):
                first_stage, last_stage = 0, stage_manager.num_stages - 1
                return [
                    {
                        first_stage: module.transformer.wte.weight,
                        last_stage: module.lm_head.weight,
                    }
                ]
        return []


# GPT2ForQuestionAnswering
class GPT2ForQuestionAnsweringPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("qa_outputs")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "qa_outputs" not in template.modules_per_stage[-1]:
            raise ValueError("qa_outputs must be in the last stage.")

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForQuestionAnswering

        module_policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForQuestionAnswering,
                new_forward=GPT2PipelineForwards.gpt2_for_question_answering_forward,
                policy=module_policy,
            )

        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared_params in gpt2 for QA."""
        return []


# GPT2ForTokenClassification
class GPT2ForTokenClassificationPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.extend(["dropout", "classifier"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["dropout", "classifier"]
        ):
            raise ValueError("dropout and classifier must be in the last stage.")

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForTokenClassification

        module_policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                GPT2ForTokenClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            module_policy.update(addon_module)

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForTokenClassification,
                new_forward=GPT2PipelineForwards.gpt2_for_token_classification_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []


# GPT2ForSequenceClassification
class GPT2ForSequenceClassificationPolicy(GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("score")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "score" not in template.modules_per_stage[-1]:
            raise ValueError("score must be in the last stage.")

    def module_policy(self):
        from transformers.models.gpt2.modeling_gpt2 import GPT2ForSequenceClassification

        module_policy = super().module_policy()

        if self.pipeline_stage_manager is not None:
            self.set_pipeline_forward(
                model_cls=GPT2ForSequenceClassification,
                new_forward=GPT2PipelineForwards.gpt2_for_sequence_classification_forward,
                policy=module_policy,
            )
        return module_policy

    def get_held_layers(self) -> List[nn.Module]:
        held_layers = super().get_held_layers()
        if self.pipeline_stage_manager.is_last_stage():
            held_layers.append(self.model.score)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in GPT2ForTokenClassification."""
        return []
