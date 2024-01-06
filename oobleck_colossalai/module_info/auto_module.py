import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import torch.nn as nn
from colossalai.shardformer.policies.auto_policy import _POLICY_LIST, _fullname

from loguru import logger


class BaseModuleInfo(ABC):
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def modules(self) -> list[str]:
        pass


@dataclass
class ModuleInfoLocation:
    file_name: str
    class_name: str


_MODULE_LIST = {
    package: ModuleInfoLocation(model, package.split(".")[-1])
    for model in ["bert", "llama", "t5", "gpt2", "vit", "opt", "bloom"]
    for package in _POLICY_LIST.keys()
    if model in package
}


def get_module_names(model: nn.Module) -> list[str]:
    full_name = _fullname(model)
    module_info_location = _MODULE_LIST.get(full_name, None)

    if module_info_location is None:
        raise ValueError(
            f"Module Info for {model.__class__.__qualname__} is not registered."
        )
    else:
        module_name = f"oobleck_colossalai.module_info.{module_info_location.file_name}"
        module = importlib.import_module(module_name)
        class_name = getattr(module, module_info_location.class_name)
        module_names: Type[BaseModuleInfo] = class_name(model).modules()

        logger.debug(f"Module names for {model.__class__.__qualname__}: {module_names}")
        return module_names
