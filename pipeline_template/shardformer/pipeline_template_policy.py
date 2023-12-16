import functools
from typing import Union

from colossalai.shardformer.policies.base_policy import Policy
import torch.nn as nn

from pipeline_template.pipeline_template import PipelineTemplate
from pipeline_template.stage_manager import HeterogeneousPipelineStageManager


class PipelineTemplatePolicyWrapper:
    """A wrapper class of any model policy.

    This class is used to wrap any model policy to support layer distribution
    based on the given pipeline template.

    A wrapped policy can be used as a normal policy; it is a child class of the
    given policy class.
    But its static methods (distribute_layers, get_stage_index) are overwritten
    so that pipeline template can be used to distribute layers.

    Example:
        >>> policy = ... # any policy, e.g. GPT2Policy()
        >>> wrapper = PipelineTemplatePolicyWrapper(pipeline_template)
        >>> wrapped_policy = wrapper.wrap(policy)
        >>> isinstance(wrapped_policy, type(policy))
        True
    """

    def __init__(self, pipeline_template: PipelineTemplate):
        self.pipeline_template = pipeline_template

    def get_layers_from_template(self, policy: Policy) -> list[nn.Module]:
        """Get layers from the given policy based on the pipeline template."""
        assert (
            policy.pipeline_stage_manager is not None
            and policy.pipeline_stage_manager.num_stages
            == len(self.pipeline_template.modules_per_stage)
        ), "The given policy is not compatible with the given pipeline template."
        return [
            policy.model.get_submodule(name)
            for name in self.pipeline_template.modules_per_stage[
                policy.pipeline_stage_manager.stage
            ]
        ]

    def wrap(self, policy: Policy):
        """
        Wrap the given policy to support layer distribution
        based on the given pipeline template.

        Args:
            policy (Policy): any policy to be wrapped

        Returns:
            PipelineTemplatePolicy: a wrapped policy.
                Can be used as a normal policy instance,
                but its static methods (distribute_layers, get_stage_index)
                are overwritten.
        """

        class PipelineTemplatePolicy(type(policy)):
            def __init__(self):
                super().__init__()
                self._policy = policy

            def __getattr__(self, name):
                return getattr(self._policy, name)

            def get_held_layers(self) -> list[nn.Module]:
                pass

        wrapped_policy = PipelineTemplatePolicy()
        wrapped_policy.get_held_layers = functools.partial(
            PipelineTemplatePolicyWrapper.get_layers_from_template, self, policy
        )

        return wrapped_policy
