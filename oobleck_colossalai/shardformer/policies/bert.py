# Copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/bert.py

from __future__ import annotations

import itertools
from functools import partial
from typing import Callable, Dict, List, cast

import colossalai.shardformer.layer as col_nn
import torch.nn as nn
from colossalai.shardformer.modeling.bert import (
    BertPipelineForwards,
    bert_sequence_parallel_forward_fn,
    get_bert_flash_attention_forward,
    get_jit_fused_bert_output_forward,
    get_jit_fused_bert_self_output_forward,
)
from colossalai.shardformer.modeling.jit import get_jit_fused_dropout_add_func
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
from torch import Tensor
from torch.nn import Module
from transformers import BertConfig, PretrainedConfig

__all__ = [
    "BertPolicy",
    "BertModelPolicy",
    "BertForPreTrainingPolicy",
    "BertLMHeadModelPolicy",
    "BertForMaskedLMPolicy",
    "BertForNextSentencePredictionPolicy",
    "BertForSequenceClassificationPolicy",
    "BertForTokenClassificationPolicy",
    "BertForMultipleChoicePolicy",
    "BertForQuestionAnsweringPolicy",
]


class BertPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = True
    ) -> List[str]:
        assert isinstance(
            config, BertConfig
        ), "config must be an instance of BertConfig"
        config: BertConfig = cast(BertConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend([f"encoder.layer.{i}" for i in range(config.num_hidden_layers)])
        if add_pooling_layer:
            modules.append("pooler")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.bert.modeling_bert" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "BertModel" else "bert."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embeddings" not in template.modules_per_stage[0]:
            raise ValueError("The first stage must contain the embeddings module.")

        if f"{prefix}pooler" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the pooler module.")

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
        from transformers.models.bert.modeling_bert import (
            BertEmbeddings,
            BertLayer,
            BertModel,
            BertOutput,
            BertSelfAttention,
            BertSelfOutput,
        )

        policy = {}

        if self.shard_config.enable_fused_normalization:
            norm_cls = col_nn.FusedLayerNorm
        else:
            norm_cls = col_nn.LayerNorm

        use_sequence_parallel = self.shard_config.enable_sequence_parallelism
        overlap = self.shard_config.enable_sequence_overlap
        if self.shard_config.enable_tensor_parallelism:
            policy[BertLayer] = ModulePolicyDescription(
                attribute_replacement={
                    "attention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.all_head_size": self.model.config.hidden_size
                    // self.shard_config.tensor_parallel_size,
                    "attention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                    "crossattention.self.num_attention_heads": self.model.config.num_attention_heads
                    // self.shard_config.tensor_parallel_size,
                },
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="attention.self.query",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.key",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.value",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.self.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"seq_parallel": use_sequence_parallel},
                    ),
                    SubModuleReplacementDescription(
                        suffix="attention.output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                    SubModuleReplacementDescription(
                        suffix="intermediate.dense",
                        target_module=col_nn.Linear1D_Col,
                        kwargs={
                            "seq_parallel": use_sequence_parallel,
                            "overlap": overlap,
                        },
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dense",
                        target_module=col_nn.Linear1D_Row,
                        kwargs={"seq_parallel": use_sequence_parallel},
                    ),
                    SubModuleReplacementDescription(
                        suffix="output.dropout",
                        target_module=col_nn.DropoutForParallelInput,
                    ),
                ],
            )

            policy[BertEmbeddings] = ModulePolicyDescription(
                sub_module_replacement=[
                    SubModuleReplacementDescription(
                        suffix="word_embeddings",
                        target_module=col_nn.VocabParallelEmbedding1D,
                    ),
                    SubModuleReplacementDescription(
                        suffix="dropout",
                        target_module=col_nn.DropoutForReplicatedInput,
                    ),
                ]
            )

        if use_sequence_parallel:
            self.append_or_create_method_replacement(
                description={
                    "forward": bert_sequence_parallel_forward_fn(self.shard_config)
                },
                policy=policy,
                target_key=BertModel,
            )

        # optimization configuration
        # Handle bert layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="attention.output.LayerNorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
                SubModuleReplacementDescription(
                    suffix="output.LayerNorm",
                    target_module=norm_cls,
                    kwargs={"sp_partial_derived": use_sequence_parallel},
                ),
            ],
            policy=policy,
            target_key=BertLayer,
        )
        # handle embedding layer
        self.append_or_create_submodule_replacement(
            description=[
                SubModuleReplacementDescription(
                    suffix="LayerNorm",
                    target_module=norm_cls,
                )
            ],
            policy=policy,
            target_key=BertEmbeddings,
        )

        # use flash attention
        if self.shard_config.enable_flash_attention:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_bert_flash_attention_forward(),
                },
                policy=policy,
                target_key=BertSelfAttention,
            )

        # use jit operator
        if self.shard_config.enable_jit_fused:
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bert_self_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BertSelfOutput,
            )
            self.append_or_create_method_replacement(
                description={
                    "forward": get_jit_fused_bert_output_forward(),
                    "dropout_add": get_jit_fused_dropout_add_func(),
                },
                policy=policy,
                target_key=BertOutput,
            )

        return policy

    def add_lm_head_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        # optimize for tensor parallelism
        if self.shard_config.enable_tensor_parallelism:
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="decoder",
                    target_module=col_nn.Linear1D_Col,
                    kwargs={"gather_output": True},
                ),
                policy=base_policy,
                target_key=BertLMPredictionHead,
            )

        # optimize with fused normalization
        if self.shard_config.enable_fused_normalization:
            # Handle bert lm prediction head
            self.append_or_create_submodule_replacement(
                description=SubModuleReplacementDescription(
                    suffix="transform.LayerNorm",
                    target_module=col_nn.FusedLayerNorm,
                ),
                policy=base_policy,
                target_key=BertLMPredictionHead,
            )
        return base_policy

    def add_lm_prediction_policy(self, base_policy):
        from transformers.models.bert.modeling_bert import BertLMPredictionHead

        method_replacement = {
            "_save_to_state_dict": col_nn.ParallelModule._save_to_state_dict,
            "_load_from_state_dict": col_nn.ParallelModule._load_from_state_dict,
        }
        self.append_or_create_method_replacement(
            description=method_replacement,
            policy=base_policy,
            target_key=BertLMPredictionHead,
        )
        return base_policy

    def postprocess(self):
        return self.model

    def set_pipeline_forward(
        self, model_cls: nn.Module, new_forward: Callable, policy: Dict
    ) -> None:
        """
        If under pipeline parallel setting, replacing the original forward method of huggingface
        to customized forward method, and add this changing to policy.
        """
        if self.pipeline_stage_manager is None:
            return

        stage_manager = self.pipeline_stage_manager
        if self.model.__class__.__name__ == "BertModel":
            module = self.model
        else:
            module = self.model.bert

        if stage_manager.is_interleave:
            layers_per_stage = self.distribute_layers(
                len(module.encoder.layer),
                stage_manager.num_stages * stage_manager.num_model_chunks,
            )
            stage_manager.stage_indices = self.get_stage_index(
                layers_per_stage,
                stage_manager.stage,
                num_model_chunks=stage_manager.num_model_chunks,
                num_stages=stage_manager.num_stages,
            )
            method_replacement = {
                "forward": partial(
                    new_forward,
                    stage_manager=stage_manager,
                    shard_config=self.shard_config,
                )
            }

        else:
            layers_per_stage = self.distribute_layers(
                len(module.encoder.layer), stage_manager.num_stages
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

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        assert self.pipeline_stage_manager is not None

        if self.model.__class__.__name__ == "BertModel":
            module = self.model
        else:
            module = self.model.bert
        stage_manager = self.pipeline_stage_manager

        held_layers = []
        if stage_manager.is_interleave:
            assert stage_manager.num_model_chunks is not None
            layers_per_stage = self.distribute_layers(
                len(module.encoder.layer),
                stage_manager.num_stages * stage_manager.num_model_chunks,
            )
            stage_indices = self.get_stage_index(
                layers_per_stage,
                stage_manager.stage,
                num_model_chunks=stage_manager.num_model_chunks,
                num_stages=stage_manager.num_stages,
            )
            if stage_manager.is_first_stage(ignore_chunk=True):
                held_layers.append(module.embeddings)
            for start_idx, end_idx in stage_indices:
                held_layers.extend(module.encoder.layer[start_idx:end_idx])
            if stage_manager.is_last_stage(ignore_chunk=True):
                held_layers.append(module.pooler)

        else:
            layers_per_stage = self.distribute_layers(
                len(module.encoder.layer), stage_manager.num_stages
            )
            if stage_manager.is_first_stage():
                held_layers.append(module.embeddings)
            start_idx, end_idx = self.get_stage_index(
                layers_per_stage, stage_manager.stage
            )
            held_layers.extend(module.encoder.layer[start_idx:end_idx])
            if stage_manager.is_last_stage():
                held_layers.append(module.pooler)

        return held_layers


# BertModel
class BertModelPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        return BertPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertModel,
                new_forward=BertPipelineForwards.bert_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage."""
        held_layers = super().get_held_layers()
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        """No shared params in bert model"""
        return []


# BertForPreTraining
class BertForPreTrainingPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("cls")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForPreTraining

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForPreTraining,
                new_forward=BertPipelineForwards.bert_for_pretraining_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """Get pipeline layers for current stage"""
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)

        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        model = self.model
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(model.bert.embeddings.word_embeddings.weight) == id(
                model.cls.predictions.decoder.weight
            ):
                # tie weights
                return [
                    {
                        0: model.bert.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertLMHeadModel
class BertLMHeadModelPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> List[str]:
        modules = [
            f"bert.{module}"
            for module in BertPolicy.get_all_modules(config, add_pooling_layer)
        ]
        modules.append("cls")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertLMHeadModel

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertLMHeadModel,
                new_forward=BertPipelineForwards.bert_lm_head_model_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(bert_model.embeddings.word_embeddings.weight) == id(
                self.model.cls.predictions.decoder.weight
            ):
                # tie weights
                return [
                    {
                        0: bert_model.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertForMaskedLM
class BertForMaskedLMPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> List[str]:
        modules = [
            f"bert.{module}"
            for module in BertPolicy.get_all_modules(config, add_pooling_layer)
        ]
        modules.append("cls")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")

    def module_policy(self):
        policy = super().module_policy()
        policy = self.add_lm_head_policy(policy)
        policy = self.add_lm_prediction_policy(policy)
        from transformers.models.bert.modeling_bert import BertForMaskedLM

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForMaskedLM,
                new_forward=BertPipelineForwards.bert_for_masked_lm_forward,
                policy=policy,
            )
        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        bert_model = self.model.bert
        if self.pipeline_stage_manager and self.pipeline_stage_manager.num_stages > 1:
            if id(bert_model.embeddings.word_embeddings.weight) == id(
                self.model.cls.predictions.decoder.weight
            ):
                # tie weights
                return [
                    {
                        0: bert_model.embeddings.word_embeddings.weight,
                        self.pipeline_stage_manager.num_stages
                        - 1: self.model.cls.predictions.decoder.weight,
                    }
                ]
        return []


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("dropout")
        modules.append("classifier")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["dropout", "classifier"]
        ):
            raise ValueError(
                "The last stage must contain the dropout and classifier module."
            )

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForSequenceClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForSequenceClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForSequenceClassification,
                new_forward=BertPipelineForwards.bert_for_sequence_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForTokenClassification
class BertForTokenClassificationPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> List[str]:
        modules = [
            f"bert.{module}"
            for module in BertPolicy.get_all_modules(config, add_pooling_layer)
        ]
        modules.append("dropout")
        modules.append("classifier")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["dropout", "classifier"]
        ):
            raise ValueError(
                "The last stage must contain the dropout and classifier module."
            )

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForTokenClassification

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForTokenClassification: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForTokenClassification,
                new_forward=BertPipelineForwards.bert_for_token_classification_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("cls")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")

    def module_policy(self):
        policy = super().module_policy()
        from transformers.models.bert.modeling_bert import BertForNextSentencePrediction

        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForNextSentencePrediction,
                new_forward=BertPipelineForwards.bert_for_next_sentence_prediction_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.cls)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


# BertForMultipleChoice
class BertForMultipleChoicePolicy(BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> List[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("dropout")
        modules.append("classifier")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["dropout", "classifier"]
        ):
            raise ValueError(
                "The last stage must contain the dropout and classifier module."
            )

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForMultipleChoice

        policy = super().module_policy()

        if self.shard_config.enable_tensor_parallelism:
            addon_module = {
                BertForMultipleChoice: ModulePolicyDescription(
                    sub_module_replacement=[
                        SubModuleReplacementDescription(
                            suffix="dropout",
                            target_module=col_nn.DropoutForParallelInput,
                        )
                    ]
                )
            }
            policy.update(addon_module)
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForMultipleChoice,
                new_forward=BertPipelineForwards.bert_for_multiple_choice_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.dropout)
            held_layers.append(self.model.classifier)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []


class BertForQuestionAnsweringPolicy(BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> List[str]:
        modules = [
            f"bert.{module}"
            for module in BertPolicy.get_all_modules(config, add_pooling_layer)
        ]
        modules.append("qa_outputs")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "qa_outputs" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the qa_outputs module.")

    def module_policy(self):
        from transformers.models.bert.modeling_bert import BertForQuestionAnswering

        policy = super().module_policy()
        if self.pipeline_stage_manager:
            self.set_pipeline_forward(
                model_cls=BertForQuestionAnswering,
                new_forward=BertPipelineForwards.bert_for_question_answering_forward,
                policy=policy,
            )

        return policy

    def get_held_layers(self) -> List[Module]:
        """
        get pipeline layers for current stage
        """
        held_layers = super().get_held_layers()
        stage_manager = self.pipeline_stage_manager
        if stage_manager.is_last_stage(ignore_chunk=True):
            held_layers.append(self.model.qa_outputs)
        return held_layers

    def get_shared_params(self) -> List[Dict[int, Tensor]]:
        # no shared params for sequence classification model
        return []
