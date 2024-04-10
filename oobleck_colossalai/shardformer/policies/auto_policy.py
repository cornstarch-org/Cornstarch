# Code copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/auto_policy.py

import importlib
from typing import Type

from colossalai.shardformer.policies.auto_policy import (
    _POLICY_LIST,
    PolicyLocation,
)
from colossalai.shardformer.policies.base_policy import Policy

__all__ = ["get_policy_type", "get_autopolicy", "import_policy"]


def import_policy(policy_location: PolicyLocation) -> Policy:
    """
    Dynamically import a Policy class based on the policy location.
    """
    module_name = f"cornstarch.shardformer.policies.{policy_location.file_name}"
    module = importlib.import_module(module_name)
    return getattr(module, policy_location.class_name)


def get_policy_type(model_name: str) -> Type[Policy]:
    policy_location = _POLICY_LIST.get(model_name, None)
    if policy_location is None:
        raise NotImplementedError(
            f"Auto policy for {model_name} is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )

    return import_policy(policy_location)


def get_autopolicy(model_name: str) -> Policy:
    r"""
    Return the auto policy for the model

    Args:
        pipeline_template (:class:`PipelineTemplate`): The pipeline template to get the corresponding policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    return get_policy_type(model_name)()
