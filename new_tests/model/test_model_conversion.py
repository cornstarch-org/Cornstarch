from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.clip.modeling_clip import CLIPVisionModel
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel
from transformers.models.whisper.modeling_whisper import WhisperModel

from new_cornstarch.models import from_hf_config
from new_tests.model.model_configs import (
    clip_vision_config,
    deepseek_v3_config,
    llama_config,
    qwen3_5_config,
    qwen3_5_moe_config,
    qwen3_vl_vision_config,
    siglip2_vision_config,
    whisper_config,
)


ATTN_IMPLEMENTATION = "kernels-community/flash-attn3"


def _materialize_device() -> torch.device:
    """Choose CUDA when available, otherwise fall back to CPU for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_state_dict_equal(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> None:
    """Assert two state dicts contain identical keys and tensor values."""
    assert sorted(expected.keys()) == sorted(actual.keys())
    for key, expected_tensor in expected.items():
        assert torch.equal(expected_tensor.cpu(), actual[key].cpu()), key


def _model_is_meta(model: torch.nn.Module) -> bool:
    """Return whether every model tensor is still on the meta device."""
    tensors = list(model.parameters()) + list(model.buffers())
    return bool(tensors) and all(tensor.is_meta for tensor in tensors)


def _assert_cornstarch_owned_structure(
    hf_state_dict: dict[str, torch.Tensor],
    cornstarch_model: torch.nn.Module,
) -> None:
    """Assert Cornstarch owns the module tree while preserving HF export keys."""
    assert not hasattr(cornstarch_model, "hf_model")
    assert not hasattr(cornstarch_model, "_forward_owner")
    assert set(cornstarch_model.state_dict().keys()) != set(hf_state_dict.keys())


def _assert_hf_checkpoint_compatibility(
    config: PretrainedConfig,
    hf_factory: Callable[[PretrainedConfig], PreTrainedModel],
    cornstarch_factory: Callable[[PretrainedConfig], torch.nn.Module],
    hf_loader: Callable[[Path], PreTrainedModel],
    tmp_path: Path,
) -> None:
    """Verify Cornstarch preserves HF checkpoint load, export, and save behavior."""
    hf_model = hf_factory(config)
    cornstarch_model = cornstarch_factory(config)

    assert cornstarch_model.attn_implementation == ATTN_IMPLEMENTATION
    _assert_cornstarch_owned_structure(hf_model.state_dict(), cornstarch_model)

    missing, unexpected = cornstarch_model.load_hf_state_dict(hf_model.state_dict())
    assert missing == []
    assert unexpected == []

    cornstarch_model.materialize(_materialize_device())
    assert not _model_is_meta(cornstarch_model)
    _assert_state_dict_equal(hf_model.state_dict(), cornstarch_model.to_hf_state_dict())

    cornstarch_model.save_pretrained(tmp_path)
    hf_reloaded = hf_loader(tmp_path)
    _assert_state_dict_equal(cornstarch_model.to_hf_state_dict(), hf_reloaded.state_dict())


def _floating_tensors(output: object) -> list[torch.Tensor]:
    """Collect floating tensors from nested model outputs."""
    if isinstance(output, torch.Tensor):
        return [output] if output.is_floating_point() else []
    if hasattr(output, "to_tuple"):
        return _floating_tensors(output.to_tuple())
    if isinstance(output, Mapping):
        values = output.values()
    elif isinstance(output, (list, tuple)):
        values = output
    else:
        return []

    tensors: list[torch.Tensor] = []
    for value in values:
        tensors.extend(_floating_tensors(value))
    return tensors


def _assert_outputs_close(expected: object, actual: object) -> None:
    """Assert BF16 model outputs are numerically close."""
    expected_tensors = _floating_tensors(expected)
    actual_tensors = _floating_tensors(actual)
    assert len(expected_tensors) == len(actual_tensors)
    for expected_tensor, actual_tensor in zip(expected_tensors, actual_tensors, strict=True):
        assert expected_tensor.shape == actual_tensor.shape
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            rtol=1e-5,
            atol=1e-5,
        )


def _move_inputs(inputs: dict[str, object], device: torch.device) -> dict[str, object]:
    """Move generated integrity-test inputs to the selected device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _language_inputs(config: PretrainedConfig) -> dict[str, object]:
    """Build deterministic token inputs for causal language models."""
    input_ids = torch.arange(6, dtype=torch.long).unsqueeze(0) % config.vocab_size
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _clip_inputs(config: PretrainedConfig) -> dict[str, object]:
    """Build BF16 image inputs for CLIP-style vision encoders."""
    image_size = int(config.image_size)
    num_channels = int(getattr(config, "num_channels", 3))
    return {
        "pixel_values": torch.randn(
            1,
            num_channels,
            image_size,
            image_size,
            dtype=torch.bfloat16,
        )
    }


def _siglip2_inputs(config: PretrainedConfig) -> dict[str, object]:
    """Build patchified BF16 image inputs for SigLIP2 NaFlex encoders."""
    patch_size = int(config.patch_size)
    num_channels = int(getattr(config, "num_channels", 3))
    spatial_shapes = torch.tensor([[2, 2]], dtype=torch.long)
    num_patches = int(spatial_shapes.prod().item())
    return {
        "pixel_values": torch.randn(
            1,
            num_patches,
            num_channels * patch_size * patch_size,
            dtype=torch.bfloat16,
        ),
        "pixel_attention_mask": torch.ones(1, num_patches, dtype=torch.bool),
        "spatial_shapes": spatial_shapes,
    }


def _qwen3_vl_inputs(config: PretrainedConfig) -> dict[str, object]:
    """Build patchified BF16 inputs for Qwen3-VL vision encoders."""
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    num_patches = int(grid_thw.prod().item())
    patch_width = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    return {
        "hidden_states": torch.randn(num_patches, patch_width, dtype=torch.bfloat16),
        "grid_thw": grid_thw,
    }


def _whisper_inputs(config: PretrainedConfig) -> dict[str, object]:
    """Build BF16 audio features for Whisper encoder-decoder models."""
    decoder_start_token_id = config.decoder_start_token_id or config.bos_token_id or 0
    return {
        "input_features": torch.randn(
            1,
            config.num_mel_bins,
            config.max_source_positions * 2,
            dtype=torch.bfloat16,
        ),
        "decoder_input_ids": torch.tensor([[decoder_start_token_id]], dtype=torch.long),
        "use_cache": False,
    }


MODEL_CASES = [
    pytest.param(
        llama_config,
        lambda cfg: AutoModelForCausalLM.from_config(cfg),
        lambda path: AutoModelForCausalLM.from_pretrained(path),
        _language_inputs,
        id="llama",
    ),
    pytest.param(
        qwen3_5_config,
        lambda cfg: Qwen3_5ForCausalLM(cfg),
        lambda path: Qwen3_5ForCausalLM.from_pretrained(path),
        _language_inputs,
        id="qwen3_5",
    ),
    pytest.param(
        qwen3_5_moe_config,
        lambda cfg: Qwen3_5MoeForCausalLM(cfg),
        lambda path: Qwen3_5MoeForCausalLM.from_pretrained(path),
        _language_inputs,
        id="qwen3_5_moe",
    ),
    pytest.param(
        deepseek_v3_config,
        lambda cfg: DeepseekV3ForCausalLM(cfg),
        lambda path: DeepseekV3ForCausalLM.from_pretrained(path),
        _language_inputs,
        id="deepseek_v3",
    ),
    pytest.param(
        clip_vision_config,
        lambda cfg: CLIPVisionModel(cfg),
        lambda path: CLIPVisionModel.from_pretrained(path),
        _clip_inputs,
        id="clip",
    ),
    pytest.param(
        siglip2_vision_config,
        lambda cfg: Siglip2VisionModel(cfg),
        lambda path: Siglip2VisionModel.from_pretrained(path),
        _siglip2_inputs,
        id="siglip2",
    ),
    pytest.param(
        qwen3_vl_vision_config,
        lambda cfg: Qwen3VLVisionModel(cfg),
        lambda path: Qwen3VLVisionModel.from_pretrained(path),
        _qwen3_vl_inputs,
        id="qwen3_vl",
    ),
    pytest.param(
        whisper_config,
        lambda cfg: WhisperModel(cfg),
        lambda path: WhisperModel.from_pretrained(path),
        _whisper_inputs,
        id="whisper",
    ),
]


def _assert_integrity(
    config: PretrainedConfig,
    hf_factory: Callable[[PretrainedConfig], PreTrainedModel],
    input_factory: Callable[[PretrainedConfig], dict[str, object]],
) -> None:
    """Run matching HF and Cornstarch models in BF16 and compare outputs."""
    torch.manual_seed(0)
    device = _materialize_device()
    hf_model = hf_factory(config).to(device=device, dtype=torch.bfloat16).eval()
    cornstarch_model = from_hf_config(config, attn_implementation=ATTN_IMPLEMENTATION)
    missing, unexpected = cornstarch_model.load_hf_state_dict(hf_model.state_dict())
    assert missing == []
    assert unexpected == []
    cornstarch_model.materialize(device).eval()

    inputs = _move_inputs(input_factory(config), device)
    with torch.no_grad():
        expected = hf_model(**inputs)
        actual = cornstarch_model(**inputs)
    _assert_outputs_close(expected, actual)
    if "labels" in inputs:
        assert getattr(actual, "loss", None) is not None
        assert getattr(expected, "loss", None) is not None


@pytest.mark.parametrize(
    ("config_factory", "hf_factory", "_hf_loader", "input_factory"),
    MODEL_CASES,
)
def test_converted_model_integrity(
    config_factory: Callable[[], PretrainedConfig],
    hf_factory: Callable[[PretrainedConfig], PreTrainedModel],
    _hf_loader: Callable[[Path], PreTrainedModel],
    input_factory: Callable[[PretrainedConfig], dict[str, object]],
) -> None:
    """Check Cornstarch and Hugging Face BF16 forwards produce similar outputs."""
    _assert_integrity(
        config=config_factory(),
        hf_factory=hf_factory,
        input_factory=input_factory,
    )


@pytest.mark.parametrize(
    ("config_factory", "hf_factory", "hf_loader", "_input_factory"),
    MODEL_CASES,
)
def test_model_hf_checkpoint_compatibility(
    config_factory: Callable[[], PretrainedConfig],
    hf_factory: Callable[[PretrainedConfig], PreTrainedModel],
    hf_loader: Callable[[Path], PreTrainedModel],
    _input_factory: Callable[[PretrainedConfig], dict[str, object]],
    tmp_path: Path,
) -> None:
    """Check every supported conversion preserves HF state dicts and saving."""
    _assert_hf_checkpoint_compatibility(
        config=config_factory(),
        hf_factory=hf_factory,
        cornstarch_factory=lambda cfg: from_hf_config(cfg, attn_implementation=ATTN_IMPLEMENTATION),
        hf_loader=hf_loader,
        tmp_path=tmp_path,
    )
