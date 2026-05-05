from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel
from transformers.models.whisper.modeling_whisper import WhisperModel

from new_cornstarch.models import from_hf_config


ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"


def _materialize_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_all_meta(model: torch.nn.Module) -> None:
    tensors = list(model.parameters()) + list(model.buffers())
    assert tensors
    assert all(tensor.is_meta for tensor in tensors)


def _assert_state_dict_equal(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> None:
    assert sorted(expected.keys()) == sorted(actual.keys())
    for key, expected_tensor in expected.items():
        assert torch.equal(expected_tensor.cpu(), actual[key].cpu()), key


def _roundtrip(
    config: PretrainedConfig,
    hf_factory: Callable[[PretrainedConfig], PreTrainedModel],
    cornstarch_factory: Callable[[PretrainedConfig], torch.nn.Module],
    hf_loader: Callable[[Path], PreTrainedModel],
    tmp_path: Path,
) -> None:
    hf_model = hf_factory(config)
    cornstarch_model = cornstarch_factory(config)

    _assert_all_meta(cornstarch_model)
    assert cornstarch_model.attn_implementation == ATTN_IMPLEMENTATION

    missing, unexpected = cornstarch_model.load_hf_state_dict(hf_model.state_dict())
    assert missing == []
    assert unexpected == []
    _assert_all_meta(cornstarch_model)

    cornstarch_model.materialize(_materialize_device())
    assert not cornstarch_model.is_meta
    _assert_state_dict_equal(hf_model.state_dict(), cornstarch_model.to_hf_state_dict())

    cornstarch_model.save_pretrained(tmp_path)
    hf_reloaded = hf_loader(tmp_path)
    _assert_state_dict_equal(cornstarch_model.to_hf_state_dict(), hf_reloaded.state_dict())


@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-LlamaForCausalLM"])
def test_language_model_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name)

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: AutoModelForCausalLM.from_config(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: AutoModelForCausalLM.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration"])
def test_qwen3_5_language_model_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name).text_config

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: Qwen3_5ForCausalLM(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: Qwen3_5ForCausalLM.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["trl-internal-testing/tiny-DeepseekV3ForCausalLM"])
def test_deepseek_v3_language_model_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name)

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: DeepseekV3ForCausalLM(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: DeepseekV3ForCausalLM.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-CLIPModel"])
def test_vision_encoder_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name).vision_config

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: CLIPVisionModel(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: CLIPVisionModel.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["google/siglip2-base-patch16-naflex"])
def test_siglip2_vision_encoder_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name).vision_config
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: Siglip2VisionModel(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: Siglip2VisionModel.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["tiny-random/qwen3-vl"])
def test_qwen3_vl_vision_encoder_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name).vision_config

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: Qwen3VLVisionModel(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: Qwen3VLVisionModel.from_pretrained(path),
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize("model_name", ["hf-internal-testing/tiny-random-WhisperModel"])
def test_audio_encoder_hf_roundtrip(model_name: str, tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(model_name)

    _roundtrip(
        config=config,
        hf_factory=lambda cfg: WhisperModel(cfg),
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=lambda path: WhisperModel.from_pretrained(path),
        tmp_path=tmp_path,
    )
