from __future__ import annotations

from unittest.mock import Mock, call

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from cornstarch.models import from_hf_config
from cornstarch.models.multimodal import (
    CornstarchEncoderToLanguageProjectorConfig,
    CornstarchExecutionPlan,
    CornstarchModalityEncoder,
    CornstarchProjector,
    ExecutionFuture,
)
from tests.model.model_configs import clip_vision_config, llama_config


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Multimodal tests require CUDA and bfloat16.",
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def _materialized_language_model() -> torch.nn.Module:
    model = from_hf_config(llama_config())
    model.set_random_init()
    model.materialize(DEVICE)
    return model.to(dtype=DTYPE)


def _randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device=DEVICE, dtype=DTYPE)


def _long_tensor(data: list[list[int]]) -> torch.Tensor:
    return torch.tensor(data, device=DEVICE, dtype=torch.long)


class _MockableVisionModule(torch.nn.Module):
    modality = "vision"

    def __init__(self, features: torch.Tensor):
        super().__init__()
        self.features = features

    def forward(self, pixel_values: torch.Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.features)


class _MockableAudioModule(torch.nn.Module):
    modality = "audio"

    def __init__(self, features: torch.Tensor):
        super().__init__()
        self.features = features

    def forward(self, input_features: torch.Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.features)


class _MockableLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.pre_decoder = torch.nn.ModuleDict(
            {"embed_tokens": torch.nn.Embedding(vocab_size, hidden_size)}
        )
        self.pre_decoder["embed_tokens"].weight.data.fill_(1)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> BaseModelOutput:
        del input_ids, attention_mask, labels
        return BaseModelOutput(last_hidden_state=inputs_embeds)


def test_linear_projector_returns_expected_shape() -> None:
    config = CornstarchEncoderToLanguageProjectorConfig(
        projector_type="linear",
        in_features=4,
        out_features=6,
    )
    projector = CornstarchProjector(config).to(device=DEVICE, dtype=DTYPE)

    output = projector(_randn(2, 3, 4))

    assert output.last_hidden_state.shape == (2, 3, 6)
    assert output.last_hidden_state.device.type == "cuda"
    assert output.last_hidden_state.dtype == DTYPE


def test_mlp_projector_uses_configured_hidden_size() -> None:
    config = CornstarchEncoderToLanguageProjectorConfig(
        projector_type="mlp",
        in_features=4,
        hidden_features=5,
        out_features=6,
        activation="gelu",
    )
    projector = CornstarchProjector(config).to(device=DEVICE, dtype=DTYPE)

    output = projector(_randn(2, 3, 4))

    assert projector.projection.in_proj.out_features == 5
    assert output.last_hidden_state.shape == (2, 3, 6)
    assert output.last_hidden_state.device.type == "cuda"
    assert output.last_hidden_state.dtype == DTYPE


def test_qformer_projector_returns_query_token_features() -> None:
    config = CornstarchEncoderToLanguageProjectorConfig(
        projector_type="qformer",
        in_features=4,
        qformer_hidden_size=8,
        out_features=6,
        num_query_tokens=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
    )
    projector = CornstarchProjector(config).to(device=DEVICE, dtype=DTYPE)

    output = projector(_randn(2, 3, 4))

    assert output.last_hidden_state.shape == (2, 2, 6)
    assert output.last_hidden_state.device.type == "cuda"
    assert output.last_hidden_state.dtype == DTYPE


def test_projector_rejects_invalid_activation() -> None:
    with pytest.raises(ValueError, match="Unsupported activation"):
        CornstarchEncoderToLanguageProjectorConfig(
            projector_type="mlp",
            in_features=4,
            out_features=6,
            activation="not-an-activation",
        )


def test_projector_config_can_be_built_from_encoder_and_language_configs() -> None:
    encoder_config = clip_vision_config()
    language_config = llama_config()

    config = CornstarchEncoderToLanguageProjectorConfig.from_encoder_and_language_configs(
        encoder_config,
        language_config,
        projector_type="mlp",
    )

    assert config.in_features == encoder_config.hidden_size
    assert config.out_features == language_config.hidden_size
    assert config.projector_type == "mlp"


def test_execution_plan_detects_cycles_and_renders_graph() -> None:
    plan = CornstarchExecutionPlan()
    language_model = torch.nn.Identity()
    plan.run_language_model(
        module=language_model,
        name="first",
        inputs=ExecutionFuture("second"),
    )
    plan.run_language_model(
        module=language_model,
        name="second",
        inputs=ExecutionFuture("first"),
    )

    with pytest.raises(ValueError, match="cycle"):
        plan.validate()

    graph = plan.to_mermaid()
    assert "flowchart LR" in graph
    assert "first" in graph
    assert "second" in graph


def test_execution_plan_generates_default_names() -> None:
    plan = CornstarchExecutionPlan()
    module = torch.nn.Identity()
    module.modality = "vision"
    vision_outputs = plan.run_modality_encoder(
        module=module,
        pixel_values=torch.empty(1, device=DEVICE, dtype=DTYPE),
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=torch.nn.Identity(),
        input_ids=torch.empty(1, 1, device=DEVICE, dtype=torch.long),
        labels=torch.empty(1, 1, device=DEVICE, dtype=torch.long),
        modality_token_ids={"vision": 99},
        encoder_outputs={"vision": vision_outputs},
    )
    plan.run_language_model(module=torch.nn.Identity(), inputs=merged)

    assert [node.name for node in plan.nodes] == [
        "vision_encoder_outputs",
        "merged_language_inputs",
        "language_outputs",
    ]
    assert plan.nodes[1].dependencies == {
        "vision_encoder_outputs",
    }


def test_execution_plan_executes_named_intermediates() -> None:
    language_model = _materialized_language_model()
    token_id = language_model.config.vocab_size + 1
    input_ids = _long_tensor([[1, token_id, token_id, 2]])
    labels = input_ids.clone()
    features = BaseModelOutput(
        last_hidden_state=_randn(2, language_model.config.hidden_size)
    )
    plan = CornstarchExecutionPlan()
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=input_ids,
        labels=labels,
        encoder_outputs={"vision": features},
        modality_token_ids={"vision": token_id},
    )
    language_output = plan.run_language_model(module=language_model, inputs=merged)

    output = language_output.execute()

    assert language_output.name == "language_outputs"
    assert output.logits.shape == (1, 4, language_model.config.vocab_size)
    assert output.logits.device.type == "cuda"
    assert output.logits.dtype == DTYPE


def test_future_execute_runs_only_dependencies() -> None:
    pixel_values = _randn(1, 3, 2, 2)
    vision_features = _randn(2, 4)
    vision_module = _MockableVisionModule(vision_features)
    language_model = torch.nn.Identity()
    vision_forward = Mock(side_effect=vision_module.forward)
    language_forward = Mock(side_effect=language_model.forward)
    vision_module.forward = vision_forward
    language_model.forward = language_forward

    plan = CornstarchExecutionPlan()
    vision_output = plan.run_modality_encoder(
        module=vision_module,
        pixel_values=pixel_values,
    )
    plan.run_language_model(
        module=language_model,
        inputs={
            "inputs_embeds": vision_output,
            "attention_mask": torch.ones(1, 2, device=DEVICE, dtype=torch.bool),
            "labels": _long_tensor([[1, 2]]),
        },
    )

    output = vision_output.execute()

    vision_forward.assert_called_once_with(pixel_values=pixel_values)
    language_forward.assert_not_called()
    assert output.last_hidden_state is vision_features


def test_execution_plan_routes_data_in_dependency_order() -> None:
    pixel_values = _randn(1, 3, 2, 2)
    token_id = 9
    input_ids = _long_tensor([[1, token_id, token_id, 2]])
    labels = input_ids.clone()
    vision_features = torch.full(
        (2, 4),
        42,
        device=DEVICE,
        dtype=DTYPE,
    )
    vision_module = _MockableVisionModule(vision_features)
    language_model = _MockableLanguageModel(vocab_size=16, hidden_size=4).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    tracker = Mock()
    vision_forward = Mock(side_effect=vision_module.forward)
    language_forward = Mock(side_effect=language_model.forward)
    tracker.attach_mock(vision_forward, "vision")
    tracker.attach_mock(language_forward, "language")
    vision_module.forward = vision_forward
    language_model.forward = language_forward

    plan = CornstarchExecutionPlan()
    vision_future = ExecutionFuture("vision_encoder_outputs")
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=input_ids,
        labels=labels,
        modality_token_ids={"vision": token_id},
        encoder_outputs={"vision": vision_future},
    )
    language_output = plan.run_language_model(module=language_model, inputs=merged)
    plan.run_modality_encoder(
        module=vision_module,
        name=vision_future.name,
        pixel_values=pixel_values,
    )

    output = language_output.execute()

    language_kwargs = language_forward.call_args.kwargs
    assert tracker.mock_calls == [
        call.vision(pixel_values=pixel_values),
        call.language(
            input_ids=None,
            inputs_embeds=language_kwargs["inputs_embeds"],
            attention_mask=language_kwargs["attention_mask"],
            labels=language_kwargs["labels"],
        ),
    ]

    assert output.last_hidden_state is language_kwargs["inputs_embeds"]
    assert vision_forward.call_args.kwargs["pixel_values"] is pixel_values
    assert language_kwargs["input_ids"] is None
    assert language_kwargs["attention_mask"].shape == input_ids.shape
    assert language_kwargs["labels"][0, 1].item() == -100
    assert language_kwargs["labels"][0, 2].item() == -100

    merged_embeds = language_kwargs["inputs_embeds"]
    assert torch.equal(merged_embeds[0, 0], torch.ones(4, device=DEVICE, dtype=DTYPE))
    assert torch.equal(merged_embeds[0, 1], vision_features[0])
    assert torch.equal(merged_embeds[0, 2], vision_features[1])
    assert torch.equal(merged_embeds[0, 3], torch.ones(4, device=DEVICE, dtype=DTYPE))


def test_execution_plan_rejects_count_mismatch() -> None:
    language_model = _materialized_language_model()
    token_id = language_model.config.vocab_size + 1
    input_ids = _long_tensor([[1, token_id, token_id, 2]])
    labels = input_ids.clone()
    features = BaseModelOutput(
        last_hidden_state=_randn(1, language_model.config.hidden_size)
    )

    with pytest.raises(ValueError, match="must equal"):
        plan = CornstarchExecutionPlan()
        merged = plan.merge_modality_encoder_outputs(
            language_model=language_model,
            input_ids=input_ids,
            labels=labels,
            encoder_outputs={"vision": features},
            modality_token_ids={"vision": token_id},
        )
        merged.execute()


def test_execution_plan_vlm() -> None:
    language_model = _materialized_language_model()
    vision_encoder = from_hf_config(clip_vision_config(), model_kind="vision")
    vision_module = CornstarchModalityEncoder.from_encoder_and_language_model(
        vision_encoder,
        language_model,
        modality="vision",
    )
    vision_module.set_random_init()
    vision_module.materialize(DEVICE).to(dtype=DTYPE)
    token_id = language_model.config.vocab_size + 1
    image_size = int(vision_encoder.config.image_size)
    pixel_values = _randn(1, 3, image_size, image_size)
    num_vision_tokens = (image_size // vision_encoder.config.patch_size) ** 2 + 1
    input_ids = _long_tensor([[1] + [token_id] * num_vision_tokens + [2]])
    labels = input_ids.clone()
    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=vision_module,
        pixel_values=pixel_values,
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=input_ids,
        labels=labels,
        modality_token_ids={"vision": token_id},
        encoder_outputs={"vision": vision_outputs},
    )
    language_output = plan.run_language_model(module=language_model, inputs=merged)

    output = language_output.execute()

    assert output.logits.shape[:2] == input_ids.shape
    assert output.logits.device.type == "cuda"
    assert output.logits.dtype == DTYPE


def test_execution_plan_vlam() -> None:
    vision_token_id = 9
    audio_token_id = 10
    input_ids = _long_tensor([[1, vision_token_id, audio_token_id, audio_token_id, 2]])
    labels = input_ids.clone()
    pixel_values = _randn(1, 3, 2, 2)
    input_features = _randn(1, 8, 4)
    vision_features = torch.full((1, 4), 21, device=DEVICE, dtype=DTYPE)
    audio_features = torch.full((2, 4), 84, device=DEVICE, dtype=DTYPE)
    vision_module = _MockableVisionModule(vision_features)
    audio_module = _MockableAudioModule(audio_features)
    language_model = _MockableLanguageModel(vocab_size=16, hidden_size=4).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    vision_forward = Mock(side_effect=vision_module.forward)
    audio_forward = Mock(side_effect=audio_module.forward)
    language_forward = Mock(side_effect=language_model.forward)
    vision_module.forward = vision_forward
    audio_module.forward = audio_forward
    language_model.forward = language_forward

    plan = CornstarchExecutionPlan()
    vision_outputs = plan.run_modality_encoder(
        module=vision_module,
        pixel_values=pixel_values,
    )
    audio_outputs = plan.run_modality_encoder(
        module=audio_module,
        input_features=input_features,
    )
    merged = plan.merge_modality_encoder_outputs(
        language_model=language_model,
        input_ids=input_ids,
        labels=labels,
        modality_token_ids={"vision": vision_token_id, "audio": audio_token_id},
        encoder_outputs={"vision": vision_outputs, "audio": audio_outputs},
    )
    language_outputs = plan.run_language_model(module=language_model, inputs=merged)

    output = language_outputs.execute()

    vision_forward.assert_called_once_with(pixel_values=pixel_values)
    audio_forward.assert_called_once_with(input_features=input_features)
    language_forward.assert_called_once()
    language_kwargs = language_forward.call_args.kwargs
    assert output.last_hidden_state is language_kwargs["inputs_embeds"]
    assert language_kwargs["attention_mask"].shape == input_ids.shape
    assert language_kwargs["labels"][0, 1].item() == -100
    assert language_kwargs["labels"][0, 2].item() == -100
    assert language_kwargs["labels"][0, 3].item() == -100

    merged_embeds = language_kwargs["inputs_embeds"]
    assert torch.equal(merged_embeds[0, 0], torch.ones(4, device=DEVICE, dtype=DTYPE))
    assert torch.equal(merged_embeds[0, 1], vision_features[0])
    assert torch.equal(merged_embeds[0, 2], audio_features[0])
    assert torch.equal(merged_embeds[0, 3], audio_features[1])
    assert torch.equal(merged_embeds[0, 4], torch.ones(4, device=DEVICE, dtype=DTYPE))
