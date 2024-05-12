from __future__ import annotations

import itertools
from typing import cast

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.llama import (
    LlamaForCausalLMPolicy as ColossalLlamaForCausalLMPolicy,
)
from colossalai.shardformer.policies.llama import (
    LlamaForSequenceClassificationPolicy as ColossalLlamaForSequenceClassificationPolicy,
)
from colossalai.shardformer.policies.llama import (
    LlamaModelPolicy as ColossalLlamaModelPolicy,
)
from transformers import LlamaConfig, PretrainedConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

__all__ = [
    "LlamaPolicy",
    "LlamaForCausalLMPolicy",
    "LlamaForSequenceClassificationPolicy",
]


class LlamaPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(
            config, LlamaConfig
        ), "config must be an instance of LlamaConfig"
        config: LlamaConfig = cast(LlamaConfig, config)

        modules = []
        modules.append("embed_tokens")
        modules.extend([f"layers.{i}" for i in range(config.num_hidden_layers)])
        modules.append("norm")

        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.llama.modeling_llama" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "LlamaModel" else "model."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embed_tokens" not in modules_in_template[0]:
            raise ValueError("embed_tokens must be in the first stage.")

        if f"{prefix}norm" not in modules_in_template[-1]:
            raise ValueError("norm must be in the last stage.")


class LlamaModelPolicy(ColossalLlamaModelPolicy, LlamaPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return LlamaPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)


class LlamaForCausalLMPolicy(ColossalLlamaForCausalLMPolicy, LlamaPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in LlamaPolicy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("lm_head must be in the last stage.")


class LlamaForSequenceClassificationPolicy(
    ColossalLlamaForSequenceClassificationPolicy, LlamaPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in LlamaPolicy.get_all_modules(config)]
        modules.append("score")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "score" not in template.modules_per_stage[-1]:
            raise ValueError("score must be in the last stage.")
