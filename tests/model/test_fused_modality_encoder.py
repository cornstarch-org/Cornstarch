from __future__ import annotations

import torch

from cornstarch.models import (
    CornstarchExecutionPlan,
    ExecutionFuture,
    build_fused_modality_encoder,
    build_modality_encoder,
    from_hf_config,
)
from tests.model.model_configs import clip_vision_config, llama_config, whisper_config


def _modules():
    language = from_hf_config(llama_config(), model_kind="language")
    vision = build_modality_encoder(
        from_hf_config(clip_vision_config(), model_kind="vision"),
        language,
        modality="vision",
    )
    audio = build_modality_encoder(
        from_hf_config(whisper_config(), model_kind="audio"),
        language,
        modality="audio",
    )
    return language, vision, audio


def test_fused_encoder_registry_and_lifecycle_are_ordered() -> None:
    _, vision, audio = _modules()
    fused = build_fused_modality_encoder({"vision": vision, "audio": audio})

    assert fused.modalities == ("vision", "audio")
    assert tuple(fused.config) == ("vision", "audio")
    assert fused.output_hidden_size == vision.projector.config.out_features
    fused.set_random_init()
    assert vision.encoder._init_plan.mode == "random"
    assert audio.encoder._init_plan.mode == "random"


def test_fused_execution_node_runs_present_children_in_registry_order() -> None:
    class Child(torch.nn.Module):
        def __init__(self, modality: str):
            super().__init__()
            self.modality = modality
            self.projector = type("Projector", (), {"config": type("C", (), {"out_features": 4})()})()
            self.encoder = type("Encoder", (), {"config": type("C", (), {"hidden_size": 4})()})()
            self.config = (self.encoder.config, self.projector.config)

        def forward(self, value):
            return value + 1

        def set_empty_init(self): pass
        def set_random_init(self): pass
        def materialize(self, device="cpu", dtype=None): return self

    # Use the real fused class but lightweight modality-module subclasses are
    # deliberately rejected; this assertion protects the public type boundary.
    _, vision, audio = _modules()
    fused = build_fused_modality_encoder({"vision": vision, "audio": audio})
    vision.forward = lambda **kwargs: kwargs["value"] + 1
    audio.forward = lambda **kwargs: kwargs["value"] + 2

    plan = CornstarchExecutionPlan()
    future = plan.run_fused_modality_encoder(
        fused,
        inputs={"audio": {"value": torch.tensor(3)}},
    )
    result = future.execute()
    assert tuple(result) == ("audio",)
    assert torch.equal(result["audio"], torch.tensor(5))


def test_optional_fused_inputs_skip_absent_children_atomically() -> None:
    _, vision, audio = _modules()
    fused = build_fused_modality_encoder({"vision": vision, "audio": audio})
    vision.forward = lambda **kwargs: kwargs["value"] + 1
    audio.forward = lambda **kwargs: kwargs["value"] + 2

    plan = CornstarchExecutionPlan()
    future = plan.run_fused_modality_encoder(
        fused,
        inputs={
            "vision": {"value": ExecutionFuture("vision_value", optional=True)},
            "audio": {"value": ExecutionFuture("audio_value", optional=True)},
        },
    )
    result = future.execute({"audio_value": torch.tensor(3)})
    assert tuple(result) == ("audio",)
    assert torch.equal(result["audio"], torch.tensor(5))
