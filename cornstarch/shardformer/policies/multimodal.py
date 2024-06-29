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
