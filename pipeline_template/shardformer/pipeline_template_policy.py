import functools
from typing import Union

from colossalai.shardformer.policies.base_policy import Policy

from pipeline_template.pipeline_template import PipelineTemplate


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
        assert num_stages == self.pipeline_template.num_stages
        return [
            len(
                [
                    module_name
                    for module_name in module_names
                    if module_name
                    not in self.pipeline_template.module_names_not_distributable
                ]
            )
            for module_names in self.pipeline_template.module_names_per_stage
        ]

    def get_stage_index(
        self,
        layers_per_stage: list[int],
        stage: int,
        num_model_chunks: int = 1,
        num_stages: int = 0,
    ) -> Union[tuple[int, int], list[tuple[int, int]]]:
        return Policy.get_stage_index(
            layers_per_stage, stage, num_model_chunks, num_stages
        )

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

        PipelineTemplatePolicy.distribute_layers = functools.partial(
            PipelineTemplatePolicyWrapper.distribute_layers, self
        )
        PipelineTemplatePolicy.get_stage_index = functools.partial(
            PipelineTemplatePolicyWrapper.get_stage_index, self
        )
        policy.__class__ = PipelineTemplatePolicy
        return policy
