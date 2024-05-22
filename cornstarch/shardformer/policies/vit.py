from __future__ import annotations

import itertools
from typing import cast

from colossalai.shardformer.policies.base_policy import Policy
from colossalai.shardformer.policies.vit import (
    ViTForImageClassificationPolicy as ColossalViTForImageClassificationPolicy,
)
from colossalai.shardformer.policies.vit import (
    ViTForMaskedImageModelingPolicy as ColossalViTForMaskedImageModelingPolicy,
)
from colossalai.shardformer.policies.vit import ViTModelPolicy as ColossalViTModelPolicy
from transformers import PretrainedConfig, ViTConfig

from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)

__all__ = [
    "ViTPolicy",
    "ViTModelPolicy",
    "ViTForImageClassificationPolicy",
    "ViTForMaskedImageModelingPolicy",
]


class ViTPolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        assert isinstance(config, ViTConfig), "config must be an instance of ViTConfig"
        config: ViTConfig = cast(ViTConfig, config)

        modules = []
        modules.append("embeddings")
        modules.extend([f"encoder.layer.{i}" for i in range(config.num_hidden_layers)])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        assert (
            "transformers.models.vit.modeling_vit" in template.model_name
        ), "The pipeline template is not for the model that the policy is designed for."

        prefix = "" if self.model.__class__.__name__ == "ViTModel" else "vit."

        assert hasattr(self.model, "config"), "model must have a config attribute"
        modules = self.get_all_modules(self.model.config)
        modules_in_template = list(itertools.chain(*template.modules_per_stage))
        if modules != modules_in_template:
            raise ValueError(
                "Modules in the pipeline template do not match the modules in the model."
            )

        if f"{prefix}embeddings" not in modules_in_template[0]:
            raise ValueError("embeddings must be in the first stage.")


# ViTModel
class ViTModelPolicy(ColossalViTModelPolicy, ViTPolicy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = ViTPolicy.get_all_modules(config)
        modules.extend(["layernorm", "pooler"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)

        if not all(
            module in template.modules_per_stage[-1]
            for module in ["layernorm", "pooler"]
        ):
            raise ValueError("layernorm and pooler must be in the last stage.")


# ViTForImageClassification
class ViTForImageClassificationPolicy(
    ColossalViTForImageClassificationPolicy, ViTPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"vit.{module}" for module in ViTPolicy.get_all_modules(config)]
        modules.extend(["vit.layernorm", "classifier"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["vit.layernorm", "classifier"]
        ):
            raise ValueError("layernorm and classifier must be in the last stage.")


# ViTForMaskedImageModeling
class ViTForMaskedImageModelingPolicy(
    ColossalViTForMaskedImageModelingPolicy, ViTPolicy
):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        modules = [f"vit.{module}" for module in ViTPolicy.get_all_modules(config)]
        modules.extend(["vit.layernorm", "decoder"])
        return modules

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        super().pipeline_template_sanity_check(template)
        if not all(
            module in template.modules_per_stage[-1]
            for module in ["vit.layernorm", "decoder"]
        ):
            raise ValueError("layernorm and decoder must be in the last stage.")
