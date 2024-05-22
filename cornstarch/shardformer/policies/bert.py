from __future__ import annotations

import itertools
from typing import cast

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.bert import (
    BertForMaskedLMPolicy as ColossalBertForMaskedLMPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForMultipleChoicePolicy as ColossalBertForMultipleChoicePolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForNextSentencePredictionPolicy as ColossalBertForNextSentencePredictionPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForPreTrainingPolicy as ColossalBertForPreTrainingPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForQuestionAnsweringPolicy as ColossalBertForQuestionAnsweringPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForSequenceClassificationPolicy as ColossalBertForSequenceClassificationPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertForTokenClassificationPolicy as ColossalBertForTokenClassificationPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertLMHeadModelPolicy as ColossalBertLMHeadModelPolicy,
)
from colossalai.shardformer.policies.bert import (
    BertModelPolicy as ColossalBertModelPolicy,
)
from transformers import BertConfig, PretrainedConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

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
    ) -> list[str]:
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


# BertModel
class BertModelPolicy(ColossalBertModelPolicy, BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return BertPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)


# BertForPreTraining
class BertForPreTrainingPolicy(ColossalBertForPreTrainingPolicy, BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("cls")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")


# BertLMHeadModel
class BertLMHeadModelPolicy(ColossalBertLMHeadModelPolicy, BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> list[str]:
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


# BertForMaskedLM
class BertForMaskedLMPolicy(ColossalBertForMaskedLMPolicy, BertPolicy):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> list[str]:
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


# BertForSequenceClassification
class BertForSequenceClassificationPolicy(
    ColossalBertForSequenceClassificationPolicy, BertPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
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


# BertForTokenClassification
class BertForTokenClassificationPolicy(
    ColossalBertForTokenClassificationPolicy, BertPolicy
):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> list[str]:
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


# BertForNextSentencePrediction
class BertForNextSentencePredictionPolicy(
    ColossalBertForNextSentencePredictionPolicy, BertPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"bert.{module}" for module in BertPolicy.get_all_modules(config)]
        modules.append("cls")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "cls" not in template.modules_per_stage[-1]:
            raise ValueError("The last stage must contain the cls module.")


# BertForMultipleChoice
class BertForMultipleChoicePolicy(ColossalBertForMultipleChoicePolicy, BertPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
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


class BertForQuestionAnsweringPolicy(
    ColossalBertForQuestionAnsweringPolicy, BertPolicy
):
    @staticmethod
    def get_all_modules(
        config: PretrainedConfig, add_pooling_layer: bool = False
    ) -> list[str]:
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
