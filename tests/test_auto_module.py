from oobleck_colossalai.module_info.auto_module import (
    _MODULE_LIST,
    get_module_names,
    ModuleInfoLocation,
)
from transformers import AutoConfig
import torch.nn as nn

import pytest
import importlib

sample_model_name = {
    "bert": "bert-base-uncased",
    "t5": "t5-small",
    "gpt2": "gpt2",
    "vit": "google/vit-base-patch16-224",
    "opt": "facebook/opt-350m",
    "bloom": "bigscience/bloom-560m",
}


@pytest.mark.parametrize("module_info", _MODULE_LIST.items(), ids=_MODULE_LIST.keys())
def test_module_load(module_info: tuple[str, ModuleInfoLocation]):
    package_path, module_info_location = module_info

    if "llama" in module_info_location.file_name:
        pytest.skip("llama is not supported yet.")

    module_path, package_name = package_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    class_name = getattr(module, package_name)

    config = AutoConfig.from_pretrained(
        sample_model_name[module_info_location.file_name]
    )
    model: nn.Module = class_name(config)
    module_names: list[str] = get_module_names(model)

    # check result of get_module_names() covers all parameters in model.
    all_params = {name for name, _ in model.named_parameters()}
    for module_name in module_names:
        try:
            submodule = model.get_submodule(module_name)
        except AttributeError as e:
            pytest.fail(f"Module {module_name} is not a submodule of {model.__class__}")

        for name, _ in submodule.named_parameters(recurse=True):
            prefixed_name = f"{module_name}.{name}" if module_name else name
            all_params.discard(prefixed_name)

    assert len(all_params) == 0
