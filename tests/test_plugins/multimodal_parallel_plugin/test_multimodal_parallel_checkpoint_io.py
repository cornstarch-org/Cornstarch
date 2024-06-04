import shutil
from pathlib import Path
from types import MethodType

import pytest
import torch
import torch.distributed as dist
from colossalai.interface import ModelWrapper, pretrained
from colossalai.lazy import LazyInitContext
from torch.testing._internal.distributed.fake_pg import FakeStore

from cornstarch.models.multimodal_language_model import MultimodalLanguageModel
from cornstarch.plugin.multimodal_parallel_plugin import MultimodalParallelCheckpointIO

language_models = [
    "meta-llama/Meta-Llama-3-8B",
    "google/gemma-1.1-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
]

vision_models = [
    "openai/clip-vit-base-patch32",
    "facebook/dinov2-base",
]


@pytest.fixture(autouse=True, scope="module")
def temp_hf_hub_dir(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    tmp_dir = tmp_path_factory.mktemp("huf")
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture(autouse=True)
def init_process_group(request: pytest.FixtureRequest):
    store = FakeStore()
    dist.init_process_group(backend="fake", store=store, rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.mark.parametrize("language_model", language_models)
@pytest.mark.parametrize("vision_model", vision_models)
def test_multimodal_parallel_checkpoint_io(
    language_model: str,
    vision_model: str,
    temp_hf_hub_dir: Path,
):
    # Create a model without lazy context for comparing
    origin_model = MultimodalLanguageModel.from_encoders_llm_pretrained(
        text_model_name_or_path=language_model,
        vision_model_name_or_path=vision_model,
        trust_remote_code=True,
        cache_dir=temp_hf_hub_dir,
    )

    # Test start
    with LazyInitContext():
        model = MultimodalLanguageModel.from_encoders_llm_pretrained(
            text_model_name_or_path=language_model,
            vision_model_name_or_path=vision_model,
            trust_remote_code=True,
            cache_dir=temp_hf_hub_dir,
        )

    pretrained_path = pretrained.get_pretrained_path(model)
    assert pretrained_path is not None and isinstance(pretrained_path, dict)
    assert list(pretrained_path.keys()) == [
        "vision_model.encoder",
        "vision_model.projector",
        "language_model",
    ]

    # Materialize it and load checkpoint
    LazyInitContext.materialize(model)

    wrapped_model = ModelWrapper(model)
    wrapped_model.update_master_params = MethodType(lambda *args: None, wrapped_model)

    checkpoint_io = MultimodalParallelCheckpointIO()
    checkpoint_io.load_model(wrapped_model, pretrained_path)

    assert len(list(model.parameters())) == len(list(origin_model.parameters()))
    params = {n: p for n, p in origin_model.named_parameters()}
    for param_name, p in model.named_parameters():
        # Skip projector as this test randomly initializes projector
        if "projector" in param_name:
            continue

        assert param_name in params
        assert torch.equal(p, params[param_name]), f"{param_name} is not equal"
