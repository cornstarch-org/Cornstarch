from __future__ import annotations

import itertools
from typing import cast

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.opt import (
    OPTForCausalLMPolicy as ColossalOPTForCausalLMPolicy,
)
from colossalai.shardformer.policies.opt import (
    OPTForQuestionAnsweringPolicy as ColossalOPTForQuestionAnsweringPolicy,
)
from colossalai.shardformer.policies.opt import (
    OPTForSequenceClassificationPolicy as ColossalOPTForSequenceClassificationPolicy,
)
from colossalai.shardformer.policies.opt import OPTModelPolicy as ColossalOPTModelPolicy
from transformers import OPTConfig, PretrainedConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

__all__ = [
    "OPTPolicy",
    "OPTModelPolicy",
    "OPTForCausalLMPolicy",
    "OPTForSequenceClassificationPolicy",
    "OPTForQuestionAnsweringPolicy",
]


class OPTPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(config, OPTConfig), "config must be an instance of OPTConfig"
        config: OPTConfig = cast(OPTConfig, config)

        modules = []
        modules.extend(["decoder.embed_tokens", "decoder.embed_positions"])
        if config.word_embed_proj_dim != config.hidden_size:
            modules.append("decoder.project_in")
        modules.extend([f"decoder.layers.{i}" for i in range(config.num_hidden_layers)])
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            modules.append("decoder.final_layer_norm")
        if config.word_embed_proj_dim != config.hidden_size:
            modules.append("decoder.project_out")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.opt.modeling_opt" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = (
            "decoder."
            if self.model.__class__.__name__ == "OPTModel"
            else "model.decoder."
        )

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if not all(
            module in template.modules_per_stage[0]
            for module in [
                f"{prefix}embed_tokens",
                f"{prefix}embed_positions",
                f"{prefix}project_in",
            ]
        ):
            raise ValueError("The embedding layers must be in the first stage.")

        if not all(
            module in template.modules_per_stage[-1]
            for module in [f"{prefix}final_layer_norm", f"{prefix}project_out"]
        ):
            raise ValueError(
                "The final layer normalization and project_out layers must be in the last stage."
            )


class OPTModelPolicy(ColossalOPTModelPolicy, OPTPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        return OPTPolicy.get_all_modules(config)

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)


class OPTForCausalLMPolicy(ColossalOPTForCausalLMPolicy, OPTPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in OPTPolicy.get_all_modules(config)]
        modules.append("lm_head")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "lm_head" not in template.modules_per_stage[-1]:
            raise ValueError("The lm_head layer must be in the last stage.")


class OPTForSequenceClassificationPolicy(
    ColossalOPTForSequenceClassificationPolicy, OPTPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in OPTPolicy.get_all_modules(config)]
        modules.append("score")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "score" not in template.modules_per_stage[-1]:
            raise ValueError("The score layer must be in the last stage.")


class OPTForQuestionAnsweringPolicy(ColossalOPTForQuestionAnsweringPolicy, OPTPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"model.{module}" for module in OPTPolicy.get_all_modules(config)]
        modules.append("qa_outputs")
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if "qa_outputs" not in template.modules_per_stage[-1]:
            raise ValueError("The qa_outputs layer must be in the last stage.")
