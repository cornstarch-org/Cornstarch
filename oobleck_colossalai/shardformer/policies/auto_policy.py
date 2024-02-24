# Code copied from https://github.com/hpcaitech/ColossalAI/blob/v0.3.5/colossalai/shardformer/policies/auto_policy.py

import importlib
from dataclasses import dataclass

import torch.nn as nn
from colossalai.shardformer.policies.auto_policy import (
    _POLICY_LIST,
    PolicyLocation,
    _fullname,
)
from colossalai.shardformer.policies.base_policy import Policy

__all__ = ["get_autopolicy", "import_policy"]


def import_policy(policy_location: PolicyLocation) -> Policy:
    """
    Dynamically import a Policy class based on the policy location.
    """
    module_name = f"oobleck_colossalai.shardformer.policies.{policy_location.file_name}"
    module = importlib.import_module(module_name)
    return getattr(module, policy_location.class_name)


def get_autopolicy(model: nn.Module) -> Policy:
    r"""
    Return the auto policy for the model

    Args:
        model (:class:`nn.Module`): The model to get the auto policy

    Return:
        :class:`Policy`: The auto policy for the model
    """
    full_name = _fullname(model)
    policy_location = _POLICY_LIST.get(full_name, None)

    if policy_location is None:
        raise NotImplementedError(
            f"Auto policy for {model.__class__.__qualname__} is not implemented\n. Supported models are {list(_POLICY_LIST.keys())}"
        )
    else:
        policy = import_policy(policy_location)
    return policy()
