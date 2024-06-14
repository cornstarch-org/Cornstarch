import pathlib
from typing import Type

import pytest
import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.clip import CLIPVisionConfig, CLIPVisionModel

from cornstarch.models.multimodal_language_model import (
    ModalModule,
    MultimodalModel,
    MultimodalProjector,
)

vision_model_name_or_path_list = [
    ("openai/clip-vit-base-patch16", CLIPVisionConfig, CLIPVisionModel),
]

language_model_name_or_path_list = [
    "meta-llama/Meta-Llama-3-8B",
]


@pytest.mark.parametrize(
    "vision_model_name_or_path", vision_model_name_or_path_list, ids=lambda x: x[0]
)
@pytest.mark.parametrize("text_model_name_or_path", language_model_name_or_path_list)
def test_load_and_save_projection(
    vision_model_name_or_path: tuple[
        str, Type[PretrainedConfig], Type[PreTrainedModel]
    ],
    text_model_name_or_path: str,
    tmp_path: pathlib.Path,
):
    # For faster testing, we do not initialize weights for the vision and language models
    vision_config = vision_model_name_or_path[1].from_pretrained(
        vision_model_name_or_path[0]
    )
    language_config = AutoConfig.from_pretrained(
        text_model_name_or_path, trust_remote_code=True
    )

    vision_model = vision_model_name_or_path[2](vision_config)

    with init_empty_weights():
        # Parameters of language models are not necessary in this test
        # Use meta device to initialize the model.
        language_model = AutoModelForCausalLM.from_config(
            language_config, trust_remote_code=True
        )

    # Initializing a projector module is done here.
    mm = MultimodalModel(
        encoders={"vision": ModalModule(vision_model)},
        language_model=language_model,
    )

    # Before saving pretrained module, there should be no files in the directory
    assert len(list(tmp_path.iterdir())) == 0

    projector = mm.encoders["vision"].projector
    assert projector is not None

    projector.config.name_or_path = (
        f"{vision_model_name_or_path[0].split('/')[-1]}-"
        f"{text_model_name_or_path.split('/')[-1]}"
    )
    projector.save_pretrained(tmp_path)

    # After saving projector module, there should be files in the directory
    assert sorted([p.name for p in tmp_path.iterdir()]) == sorted(
        ["config.json", "model.safetensors"]
    )

    vision_project_from_pretrained = MultimodalProjector.from_pretrained(tmp_path)

    # Check loaded config attributes
    assert (
        projector.config.projection_type
        == vision_project_from_pretrained.config.projection_type
    )
    assert (
        projector.config.activation == vision_project_from_pretrained.config.activation
    )
    assert (
        projector.config.in_features
        == vision_project_from_pretrained.config.in_features
    )
    assert (
        projector.config.out_features
        == vision_project_from_pretrained.config.out_features
    )
    assert (
        projector.config.encoder_model_type
        == vision_project_from_pretrained.config.encoder_model_type
    )
    assert (
        projector.config.language_model_type
        == vision_project_from_pretrained.config.language_model_type
    )

    assert sorted(projector.state_dict().keys()) == sorted(
        vision_project_from_pretrained.state_dict().keys()
    )

    for p_key in projector.state_dict().keys():
        assert torch.equal(
            projector.state_dict()[p_key],
            vision_project_from_pretrained.state_dict()[p_key],
        )
