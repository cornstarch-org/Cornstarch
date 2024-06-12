import pathlib
from typing import Type

import pytest
import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.clip import CLIPVisionConfig

from cornstarch.models.multimodal_language_model import (
    MultimodalLanguageModelProjectorConfig,
    MultimodalProjectorModel,
)

vision_model_name_or_path_list = [
    ("openai/clip-vit-base-patch16", CLIPVisionConfig),
]

language_model_name_or_path_list = [
    "meta-llama/Meta-Llama-3-8B",
]


@pytest.mark.parametrize(
    "vision_model_name_or_path", vision_model_name_or_path_list, ids=lambda x: x[0]
)
@pytest.mark.parametrize("text_model_name_or_path", language_model_name_or_path_list)
def test_load_and_save_projection(
    vision_model_name_or_path: tuple[str, Type[PretrainedConfig]],
    text_model_name_or_path: str,
    tmp_path: pathlib.Path,
):
    vision_config = vision_model_name_or_path[1].from_pretrained(
        vision_model_name_or_path[0]
    )
    language_config = AutoConfig.from_pretrained(
        text_model_name_or_path, trust_remote_code=True
    )

    projection_type = "linear"
    projector_config = MultimodalLanguageModelProjectorConfig(
        encoder_config=vision_config,
        text_config=language_config,
        projection_type=projection_type,
    )
    projector_config.name_or_path = f"{vision_model_name_or_path[0].split('/')[-1]}-{language_config.name_or_path.split('/')[-1]}-{projection_type}"

    vision_projector = MultimodalProjectorModel(projector_config)

    assert len(list(tmp_path.iterdir())) == 0

    vision_projector.save_pretrained(tmp_path)

    assert len(list(tmp_path.iterdir())) > 0

    vision_project_from_pretrained = MultimodalProjectorModel.from_pretrained(tmp_path)

    assert sorted(vision_projector.state_dict().keys()) == sorted(
        vision_project_from_pretrained.state_dict().keys()
    )

    for p_key in vision_projector.state_dict().keys():
        assert torch.equal(
            vision_projector.state_dict()[p_key],
            vision_project_from_pretrained.state_dict()[p_key],
        )
