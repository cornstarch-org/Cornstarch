from __future__ import annotations

import itertools
from typing import cast

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.gpt2 import (
    GPT2DoubleHeadsModelPolicy as ColossalGPT2DoubleHeadsModelPolicy,
)
from colossalai.shardformer.policies.gpt2 import (
    GPT2ForQuestionAnsweringPolicy as ColossalGPT2ForQuestionAnsweringPolicy,
)
from colossalai.shardformer.policies.gpt2 import (
    GPT2ForSequenceClassificationPolicy as ColossalGPT2ForSequenceClassificationPolicy,
)
from colossalai.shardformer.policies.gpt2 import (
    GPT2ForTokenClassificationPolicy as ColossalGPT2ForTokenClassificationPolicy,
)
from colossalai.shardformer.policies.gpt2 import (
    GPT2LMHeadModelPolicy as ColossalGPT2LMHeadPolicy,
)
from colossalai.shardformer.policies.gpt2 import (
    GPT2ModelPolicy as ColossalGPT2ModelPolicy,
)
from transformers import GPT2Config, PretrainedConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

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
    def get_all_modules(config: PretrainedConfig) -> list[str]:
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


# GPT2Model
class GPT2ModelPolicy(ColossalGPT2ModelPolicy, GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return GPT2Policy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)


# GPT2LMHeadModel
class GPT2LMHeadModelPolicy(ColossalGPT2LMHeadPolicy, GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")


# GPT2DoubleHeadsModel
class GPT2DoubleHeadsModelPolicy(ColossalGPT2DoubleHeadsModelPolicy, GPT2Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
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


# GPT2ForQuestionAnswering
class GPT2ForQuestionAnsweringPolicy(
    ColossalGPT2ForQuestionAnsweringPolicy, GPT2Policy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("qa_outputs")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "qa_outputs" not in template.modules_per_stage[-1]:
            raise ValueError("qa_outputs must be in the last stage.")


# GPT2ForTokenClassification
class GPT2ForTokenClassificationPolicy(
    ColossalGPT2ForTokenClassificationPolicy, GPT2Policy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
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


# GPT2ForSequenceClassification
class GPT2ForSequenceClassificationPolicy(
    ColossalGPT2ForSequenceClassificationPolicy, GPT2Policy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [
            f"transformer.{module}" for module in GPT2Policy.get_all_modules(config)
        ]
        modules.append("score")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "score" not in template.modules_per_stage[-1]:
            raise ValueError("score must be in the last stage.")
