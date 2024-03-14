# Code copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/auto_policy.py

import importlib

from colossalai.shardformer.policies.auto_policy import (
    _POLICY_LIST,
    PolicyLocation,
)
from colossalai.shardformer.policies.base_policy import Policy

from oobleck_colossalai.pipeline_template import PipelineTemplate

__all__ = ["get_autopolicy", "import_policy"]


def import_policy(policy_location: PolicyLocation) -> Policy:
    """
    Dynamically import a Policy class based on the policy location.
    """
    module_name = f"oobleck_colossalai.shardformer.policies.{policy_location.file_name}"
    module = importlib.import_module(module_name)
    return getattr(module, policy_location.class_name)


def get_autopolicy(pipeline_template: PipelineTemplate) -> Policy:
    r"""
    Return the auto policy for the model

    Args:
        pipeline_template (:class:`PipelineTemplate`): The pipeline template to get the corresponding policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    policy_location = _POLICY_LIST.get(pipeline_template.model_name, None)

    if policy_location is None:
        raise NotImplementedError(
            f"Auto policy for {pipeline_template.model_name} is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )
    else:
        policy = import_policy(policy_location)
    return policy()
