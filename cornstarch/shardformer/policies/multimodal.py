from typing import Dict, cast

from colossalai.shardformer.policies.auto_policy import _fullname
from colossalai.shardformer.policies.base_policy import ModulePolicyDescription, Policy
from torch import nn
from transformers import PretrainedConfig

from cornstarch.models.multimodal_language_model import (
    ModalModule,
    MultimodalProjectorConfig,
)
from cornstarch.pipeline_template import PipelineTemplate
from cornstarch.shardformer.policies.pipeline_template_policy import (
    PipelineTemplatePolicyBase,
)


class MultimodalProjectorPolicy(PipelineTemplatePolicyBase, Policy):
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

    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        pass

    def postprocess(self) -> nn.Module:
        return self.model


class ModalModulePolicy(Policy):
    def config_sanity_check(self):
        pass

    def preprocess(self) -> nn.Module:
        return self.model

    def module_policy(self) -> Dict[str | nn.Module, ModulePolicyDescription]:
        from cornstarch.shardformer.policies.auto_policy import get_autopolicy

        model = cast(ModalModule, self.model)
        submodules = model.get_modules()

        policies = {}
        for module in submodules:
            policy = get_autopolicy(_fullname(module))
            policy.set_model(module)
            policy.set_shard_config(self.shard_config)
            policies.update(policy.module_policy())

        return policies

    def postprocess(self) -> nn.Module:
        return self.model
