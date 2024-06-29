from typing import Dict

from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy
from torch.nn.modules import Module
from transformers import PretrainedConfig

from cornstarch.models.multimodal_language_model import MultimodalProjectorConfig
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class MultimodalProjectorPolicy(PipelineTemplatePolicyBase):
    @staticmethod
    def get_all_modules(config: MultimodalProjectorConfig) -> list[str]:
        assert isinstance(
            config, MultimodalProjectorConfig
        ), f"config must be MultimodalProjectorConfig, got {type(config)}"

        if config.projection_type == "linear":
            return ["projection"]
        elif config.projection_type == "mlp":
            return ["in_proj", "activation", "out_proj"]
        elif config.projection_type == "qformer":
            raise NotImplementedError("QFormer is not supported yet.")

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        pass


class ModalModulePolicy(PipelineTemplatePolicyBase, Policy):
    @staticmethod
    def get_all_modules(config: PretrainedConfig) -> list[str]:
        raise NotImplementedError(
            "ModalModulePolicy doesn't support `get_all_modules`."
        )

    def pipeline_template_sanity_check(self, template: PipelineTemplate):
        pass

    def config_sanity_check(self):
        pass

    def preprocess(self) -> Module:
        return self.model

    def module_policy(self) -> Dict[str | Module, ModulePolicyDescription]:
        pass

    def postprocess(self) -> Module:
        return self.model
