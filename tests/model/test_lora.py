from __future__ import annotations

import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora import LoraLayer

from cornstarch.models import attach_lora, build_modality_encoder, from_hf_config
from tests.model.model_configs import clip_vision_config, llama_config


def _lora_config() -> LoraConfig:
    return LoraConfig(target_modules=["q_proj"], r=2, lora_alpha=4)


def _lora_layers(module: torch.nn.Module) -> list[LoraLayer]:
    return [child for child in module.modules() if isinstance(child, LoraLayer)]


def test_attach_lora_mutates_materialized_module_in_place() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(4, 4))

    returned = attach_lora(
        module,
        LoraConfig(target_modules=["0"], r=2, lora_alpha=4),
    )

    assert returned is module
    assert len(_lora_layers(module)) == 1


def test_attach_lora_defers_until_cornstarch_model_materializes() -> None:
    language_model = from_hf_config(llama_config(), model_kind="language")
    language_model.set_random_init()

    returned = attach_lora(language_model, _lora_config())

    assert returned is language_model
    assert not _lora_layers(language_model)

    language_model.materialize("cpu")

    assert _lora_layers(language_model)
    assert all(not parameter.is_meta for parameter in language_model.parameters())


def test_deferred_lora_preserves_checkpoint_initialization() -> None:
    reference = from_hf_config(llama_config(), model_kind="language")
    reference.set_random_init()
    reference.materialize("cpu")
    hf_state_dict = reference.to_hf_state_dict()

    language_model = from_hf_config(llama_config(), model_kind="language")
    language_model.set_checkpoint_init(state_dict=hf_state_dict)
    attach_lora(language_model, _lora_config())
    language_model.materialize("cpu")

    adapted_projection = language_model.decoder_layers[0].self_attn.q_proj
    torch.testing.assert_close(
        adapted_projection.base_layer.weight,
        reference.decoder_layers[0].self_attn.q_proj.weight,
    )


def test_encoder_and_language_lora_are_attached_independently() -> None:
    encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    language_model = from_hf_config(llama_config(), model_kind="language")
    modality_encoder = build_modality_encoder(
        encoder, language_model, modality="vision"
    )
    modality_encoder.set_random_init()
    language_model.set_random_init()

    attach_lora(modality_encoder, _lora_config())
    modality_encoder.materialize("cpu")

    assert _lora_layers(modality_encoder.encoder)
    assert not _lora_layers(modality_encoder.projector)
    assert not _lora_layers(language_model)

    attach_lora(language_model, _lora_config())
    language_model.materialize("cpu")

    assert _lora_layers(language_model)


def test_attach_lora_rejects_duplicate_adapter_name() -> None:
    language_model = from_hf_config(llama_config(), model_kind="language")
    attach_lora(language_model, _lora_config(), adapter_name="experiment")

    with pytest.raises(ValueError, match="already attached"):
        attach_lora(language_model, _lora_config(), adapter_name="experiment")
