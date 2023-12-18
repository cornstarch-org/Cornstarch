import functools
import re

import torch.nn as nn

from colossalai.shardformer.policies.base_policy import Policy
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

    def distribute_layers(self, num_layers: int, num_stages: int) -> list[int]:
        """Distribute layers to pipeline stages based on the given pipeline template.

        Args:
            num_layers (int): the number of layers to be distributed
            num_stages (int): the number of pipeline stages

        Returns:
            list[int]: a list of layer indices per stage
        """
        num_distributed_layers = [
            sum(bool(re.search(r"\d", s)) for s in modules)
            for modules in self.pipeline_template.modules_per_stage
        ]
        assert num_layers == sum(
            num_distributed_layers
        ), "The number of layers to be distributed must be equal to the number of layers in the pipeline template."
        assert (
            num_stages == self.pipeline_template.num_stages
        ), "The number of stages must be equal to the number of stages in the pipeline template."

        return num_distributed_layers

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
            def __init__(self, policy: Policy):
                super().__init__()
                self._policy = policy

            def __getattr__(self, name):
                return getattr(self._policy, name)

        wrapped_policy = PipelineTemplatePolicy(policy)
        wrapped_policy.distribute_layers = functools.partial(
            PipelineTemplatePolicyWrapper.distribute_layers,
            self,
        )

        return wrapped_policy
