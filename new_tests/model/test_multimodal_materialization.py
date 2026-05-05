from __future__ import annotations

import torch

from new_cornstarch.models import CornstarchModalityEncoder, from_hf_config
from new_tests.model.model_configs import clip_vision_config, llama_config


def _module_tensors(module: torch.nn.Module) -> list[torch.Tensor]:
    return list(module.parameters()) + list(module.buffers())


def _assert_all_meta(module: torch.nn.Module) -> None:
    tensors = _module_tensors(module)
    assert tensors
    assert all(tensor.is_meta for tensor in tensors)


def _assert_no_meta_on_device(module: torch.nn.Module, device: torch.device) -> None:
    tensors = _module_tensors(module)
    assert tensors
    assert all(not tensor.is_meta for tensor in tensors)
    assert all(tensor.device == device for tensor in tensors)


def test_modality_encoder_factory_preserves_lazy_materialization() -> None:
    language_model = from_hf_config(llama_config(), model_kind="language")
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")

    _assert_all_meta(language_model)
    _assert_all_meta(vision_encoder)

    vision_module = CornstarchModalityEncoder.from_encoder_and_language_model(
        vision_encoder,
        language_model,
        modality="vision",
    )

    assert vision_module.encoder is vision_encoder
    _assert_all_meta(language_model)
    _assert_all_meta(vision_module.encoder)
    _assert_all_meta(vision_module.projector)

    vision_module.set_random_init()
    vision_module.materialize(torch.device("cpu"))

    _assert_all_meta(language_model)
    _assert_no_meta_on_device(vision_module, torch.device("cpu"))
