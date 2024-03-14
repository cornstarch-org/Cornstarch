from typing import Type

from colossalai.shardformer.policies.auto_policy import _fullname
from torch import nn


class PipelineTemplate:
    """A template for a single pipeline that can be used to instantiate identical pipelines."""

    @staticmethod
    def get_model_name(model: nn.Module) -> str:
        """Get the model name from the model."""
        return _fullname(model)

    @staticmethod
    def get_modules(model: nn.Module) -> list[nn.Module]:
        """Get all modules from the model."""
        # Avoid circular import
        from oobleck_colossalai.shardformer.policies.auto_policy import (
            get_policy_type,
        )
        from oobleck_colossalai.shardformer.policies.pipeline_template_policy import (
            PipelineTemplatePolicyBase,
        )

        policy: Type[PipelineTemplatePolicyBase] = get_policy_type(
            PipelineTemplate.get_model_name(model)
        )
        assert issubclass(
            policy, PipelineTemplatePolicyBase
        ), f"Policy {policy} does not inherit PipelineTemplatePolicyBase."
        return policy.get_all_modules(model.config)

    def __init__(
        self,
        model_name: str,
        modules_per_stage: list[list[str]],
        latency: float = 0.0,
        mem_required: int = 0,
    ):
        self.model_name = model_name
        self.modules_per_stage = modules_per_stage
        self.latency = latency
        self.mem_required = mem_required

    @property
    def num_layers(self) -> int:
        return sum(len(stage) for stage in self.modules_per_stage)

    @property
    def num_stages(self) -> int:
        return len(self.modules_per_stage)
